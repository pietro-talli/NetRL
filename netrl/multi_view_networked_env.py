"""
multi_view_networked_env.py
===========================
MultiViewNetworkedEnv — a gymnasium.Wrapper that simulates multiple concurrent
observers each transmitting their own copy of the environment observation
through independent (or shared) communication channels to a central node.

Motivation
----------
In many networked control scenarios a single physical plant is monitored by
several sensors or agents that each send state information over a shared or
independent wireless medium.  MultiViewNetworkedEnv extends NetworkedEnv to
N such observers while giving the caller precise per-step control over:

  * **which observers transmit** at each step (e.g. duty-cycling, failures);
  * **how many bytes** each active observer sends (e.g. variable-rate coding).

Architecture
------------
Internally the wrapper owns a single CentralNode that manages one
CommChannel + ObservationBuffer per observer.  The channel type (and whether
multiple observers share the same physical medium) is determined entirely by
the ``channel_factory`` callable:

    # Each observer gets its own independent GE channel
    factory = GEChannel

    # Each observer gets its own independent ns-3 WiFi subprocess
    factory = lambda nc: NS3WifiChannel(nc, NS3WifiConfig(distance_m=15.0))

    # All observers share ONE ns-3 infrastructure BSS (realistic contention)
    factory = make_multi_ue_wifi_factory(
        NS3WifiMultiUEConfig(n_ues=3, distances_m=[10.0, 30.0, 60.0])
    )

Observation space
-----------------
The original Box observation space is replaced with a nested Dict:

    gymnasium.spaces.Dict({
        "<observer_id>": gymnasium.spaces.Dict({
            "observations": Box(shape=(buffer_size, *obs_shape), ...),
            "recv_mask":    MultiBinary(buffer_size),
        }),
        ...
    })

The buffer for each observer is a sliding window of the last buffer_size
delivery slots.  Index [-1] is the most recent; index [0] is the oldest.
Slots where no packet arrived are zero-filled with recv_mask=False.

Step semantics
--------------
At every step the wrapper:
  1. Steps the underlying environment → raw_obs, reward, flags, info.
  2. For every observer that is active this step (see transmit_mask):
       central.receive_from(observer_id, raw_obs, step, packet_size)
  3. Flushes all observers:
       arrived_map = central.flush_and_update(step)
     This always flushes every observer's channel (even silent ones) so
     that all buffers advance by exactly one slot per step.
  4. Increments step_count.
  5. Assembles and returns the per-observer Dict observation.

The ``info`` dict returned by step() is extended with:
  ``"channel_info"``      : Dict[observer_id → channel diagnostic dict]
  ``"arrived_this_step"`` : Dict[observer_id → bool]
  ``"transmitted_this_step"`` : Dict[observer_id → bool]
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from netrl.central_node import CentralNode
from netrl.channels.comm_channel import CommChannel
from netrl.channels.network_config import NetworkConfig
from netrl.utils.multi_view_model import MultiViewModel


class MultiViewNetworkedEnv(gym.Wrapper):
    """
    Gymnasium wrapper with multiple observers and per-step transmission control.

    Parameters
    ----------
    env : gymnasium.Env
        The base environment.  Must have a ``Box`` observation space.

    config : NetworkConfig
        Shared channel and buffer configuration.  ``buffer_size`` applies to
        every observer's buffer.  For GE channels, the Markov parameters are
        also read from here.  Call ``config.validate()`` before passing.

    observer_ids : List[str]
        Unique string identifier for each observer.  The length of this list
        determines the number of independent transmission paths.  These ids
        are also used as keys in the returned observation dict and in the
        ``transmit_mask`` / ``packet_sizes`` arguments of ``step()``.

    channel_factory : Callable[[NetworkConfig], CommChannel]
        Called once per observer (in the order of observer_ids) during
        construction to create each observer's CommChannel.  Choose from:

        ``GEChannel``
            Independent Gilbert–Elliott channel per observer.  All GE
            parameters come from ``config``.

        ``PerfectChannel``
            Lossless zero-delay channel (useful for debugging).

        ``lambda nc: NS3WifiChannel(nc, NS3WifiConfig(...))``
            Independent ns-3 802.11a WiFi subprocess per observer.

        ``make_multi_ue_wifi_factory(NS3WifiMultiUEConfig(...))``
            **Shared** ns-3 infrastructure BSS for all observers.  All
            observers contend for the same wireless medium.  The factory
            must be created *before* passing it here, and
            ``NS3WifiMultiUEConfig.n_ues`` must equal ``len(observer_ids)``.

    Examples
    --------
    Three observers sharing a single 802.11a WiFi channel::

        from netrl import NetworkConfig, MultiViewNetworkedEnv
        from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory

        factory = make_multi_ue_wifi_factory(
            NS3WifiMultiUEConfig(
                n_ues=3,
                distances_m=[10.0, 30.0, 60.0],
                step_duration_ms=2.0,
            )
        )
        env = MultiViewNetworkedEnv(
            gym.make("CartPole-v1"),
            NetworkConfig(buffer_size=10),
            observer_ids=["near", "mid", "far"],
            channel_factory=factory,
        )
        obs, info = env.reset()
        # obs["near"]["observations"].shape == (10, 4)
        # obs["near"]["recv_mask"].shape    == (10,)

        # Step with all observers transmitting (default)
        obs, r, term, trunc, info = env.step(action)

        # Step: only "near" transmits, with 256 bytes
        obs, r, term, trunc, info = env.step(
            action,
            transmit_mask={"near": True, "mid": False, "far": False},
            packet_sizes={"near": 256},
        )
    """

    def __init__(
        self,
        env: gym.Env,
        config: NetworkConfig,
        observer_ids: List[str],
        multi_view_model: MultiViewModel,
        channel_factory: Callable[[NetworkConfig], CommChannel],
    ) -> None:
        super().__init__(env)
        config.validate()

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "MultiViewNetworkedEnv requires the wrapped env to have a "
                f"Box observation space, got {type(env.observation_space).__name__}."
            )
        if not observer_ids:
            raise ValueError("observer_ids must not be empty.")
        if len(observer_ids) != len(set(observer_ids)):
            raise ValueError("observer_ids must be unique.")

        self._config       = config
        self._observer_ids = list(observer_ids)
        self._step_count: int = 0

        # One channel + one buffer per observer, managed by CentralNode
        self._central = CentralNode(
            node_ids=observer_ids,
            obs_shape=multi_view_model.obs_shapes,
            obs_dtype=multi_view_model.obs_dtypes,
            config=config,
            channel_factory=channel_factory,
        )

        # Nested Dict observation space: observer → {observations, recv_mask}
        self.observation_space = spaces.Dict({
            oid: spaces.Dict({
                "observations": multi_view_model.spaces[oid],
                "recv_mask": spaces.MultiBinary(config.buffer_size),
            })
            for oid in observer_ids
        })

        self._multi_view_model = multi_view_model

    # -----------------------------------------------------------------------
    # gymnasium.Wrapper overrides
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], dict]:
        """
        Reset the wrapped environment and all channel / buffer state.

        Returns all-zeros observations with recv_mask=False for every observer
        because no transmission has occurred yet.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._central.reset()
        self._step_count = 0
        return self._build_obs(), info

    def step(
        self,
        action: Any,
        *,
        transmit_mask: Optional[Dict[str, bool]] = None,
        packet_sizes:  Optional[Dict[str, int]]  = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, bool, dict]:
        """
        Step the environment and run the multi-observer channel simulation.

        Parameters
        ----------
        action : Any
            Action compatible with the wrapped environment.

        transmit_mask : Dict[str, bool] | None, optional
            Controls which observers transmit this step.

            ``None`` (default)
                All observers transmit.
            Dict mapping observer_id → bool
                Only observers mapped to ``True`` transmit.
                Observers absent from the dict default to ``True``
                (i.e. the mask is an opt-out, not opt-in).

        packet_sizes : Dict[str, int] | None, optional
            Per-observer payload in bytes for this step.

            ``None`` (default)
                Every transmitting observer uses its channel's own default
                packet size (e.g. ``NS3WifiMultiUEConfig.packet_size_bytes``).
            Dict mapping observer_id → int
                Overrides the packet size for the named observer.
                Absent observers use the channel default.

        Returns
        -------
        obs : Dict[observer_id → Dict{"observations", "recv_mask"}]
            Observation buffers for all observers.
        reward : float
        terminated : bool
        truncated : bool
        info : dict
            Extended with:

            ``"channel_info"``
                Dict[observer_id → channel diagnostic dict from
                CommChannel.get_channel_info()].

            ``"arrived_this_step"``
                Dict[observer_id → bool] — True if a packet from that
                observer arrived at the AP during this step.

            ``"transmitted_this_step"``
                Dict[observer_id → bool] — True if that observer attempted
                to transmit this step (i.e. was not masked out).

        Notes
        -----
        ``flush_and_update`` is called for **all** observers every step,
        regardless of ``transmit_mask``.  This ensures every buffer advances
        by exactly one slot per step and that delayed packets from previous
        steps are still collected for silent observers.
        """
        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        t = self._step_count

        # Observe through the multi-view model
        multi_view_obs = self._multi_view_model.observe(self.env, raw_obs)

        # Track which observers are transmitting this step
        transmitted: Dict[str, bool] = {}
        for oid in self._observer_ids:
            active = transmit_mask is None or transmit_mask.get(oid, True)
            transmitted[oid] = active
            if active:
                pkt_size = None if packet_sizes is None else packet_sizes.get(oid)
                self._central.receive_from(oid, multi_view_obs[oid], t, pkt_size)

        # Flush ALL observers so every buffer advances uniformly.
        # Observers that did not transmit this step simply contribute None
        # to their buffer (packet not in-flight → nothing to deliver).
        arrived_map = self._central.flush_and_update(t)

        self._step_count += 1

        info["channel_info"] = {
            oid: self._central.get_channel_info(oid)
            for oid in self._observer_ids
        }
        info["arrived_this_step"] = {
            oid: arrived_map[oid] is not None
            for oid in self._observer_ids
        }
        info["transmitted_this_step"] = transmitted

        return self._build_obs(), reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble the per-observer Dict observation from CentralNode buffers."""
        return {
            oid: {"observations": obs_buf, "recv_mask": recv_mask}
            for oid, (obs_buf, recv_mask) in self._central.get_all_buffers().items()
        }

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Current step counter (0-indexed; incremented after each step call)."""
        return self._step_count

    @property
    def observer_ids(self) -> List[str]:
        """Ordered list of observer identifiers."""
        return list(self._observer_ids)

    @property
    def config(self) -> NetworkConfig:
        """The NetworkConfig used to configure this wrapper."""
        return self._config

    @property
    def central_node(self) -> CentralNode:
        """Direct access to the underlying CentralNode."""
        return self._central
