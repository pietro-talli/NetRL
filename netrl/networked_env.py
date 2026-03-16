"""
networked_env.py
================
NetworkedEnv — a gymnasium.Wrapper that simulates a noisy communication
channel between the RL agent and the environment.

At every step the wrapper:
  1. Steps the underlying environment to get the raw observation.
  2. Transmits the raw observation through the channel backend (loss + delay).
  3. Flushes the channel: collects any packet due at this step (possibly
     none if the packet was lost or not yet delivered).
  4. Updates the ObservationBuffer with the arrived packet or None.
  5. Returns the full padded buffer as the Dict observation.

Observation space override
--------------------------
The original Box(obs_shape) is replaced with:

    gymnasium.spaces.Dict({
        "observations": Box(low=-inf, high=inf,
                            shape=(buffer_size, *obs_shape),
                            dtype=obs_dtype),
        "recv_mask":    MultiBinary(buffer_size),
    })

The agent sees a sliding window of the last buffer_size delivery slots.
Slot [-1] is the most recent; slot [0] is oldest.  Slots where no packet
arrived (loss or not yet delivered) are zero-filled with recv_mask=False.

Selecting the channel backend
------------------------------
The `channel_config` parameter selects and configures the backend:

    # Gilbert-Elliott (default) — parameters come from NetworkConfig
    env = NetworkedEnv(gym.make("CartPole-v1"), network_config)

    # ns-3 802.11a WiFi — build src/ns3_wifi_sim first
    from netrl import NS3WifiConfig
    env = NetworkedEnv(
        gym.make("CartPole-v1"),
        network_config,
        channel_config=NS3WifiConfig(distance_m=15.0, step_duration_ms=2.0),
    )

The ns-3 simulation persists across steps and only resets on env.reset().
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from netrl.central_node import CentralNode
from netrl.channels.comm_channel import CommChannel, GEChannel
from netrl.channels.network_config import NetworkConfig
from netrl.channels.ns3_wifi_config import NS3WifiConfig
from netrl.channels.ns3_channel import NS3WifiChannel


class NetworkedEnv(gym.Wrapper):
    """
    Gymnasium wrapper simulating networked observation transmission.

    Parameters
    ----------
    env            : gymnasium.Env
        The base environment to wrap.  Must have a Box observation space.
    config         : NetworkConfig
        Channel and buffer configuration.  For the Gilbert-Elliott backend
        this also carries the Markov-chain and loss parameters.  Validated
        on construction.
    channel_config : NS3WifiConfig | None, optional
        Selects and configures the channel backend:

        ``None`` (default)
            Use the Gilbert-Elliott channel.  All GE parameters are taken
            from `config` (p_gb, p_bg, loss_good, loss_bad, delay_steps).

        ``NS3WifiConfig(...)``
            Use the ns-3 802.11a WiFi channel.  The binary
            ``src/ns3_wifi_sim`` must be built first
            (``bash src/build_ns3_sim.sh``).

    node_id        : str
        Identifier for this agent's transmission node.  Default "agent_0".
    """

    def __init__(
        self,
        env: gym.Env,
        config: NetworkConfig,
        channel_config: Optional[NS3WifiConfig] = None,
        node_id: str = "agent_0",
    ) -> None:
        super().__init__(env)
        config.validate()

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "NetworkedEnv requires the wrapped env to have a "
                f"Box observation space, got {type(env.observation_space)}."
            )

        self._config = config
        self._node_id = node_id
        self._step_count: int = 0

        # Resolve channel factory from channel_config
        if channel_config is None:
            channel_factory = GEChannel
        elif isinstance(channel_config, NS3WifiConfig):
            _ns3 = channel_config
            channel_factory = lambda node_cfg: NS3WifiChannel(node_cfg, _ns3)  # noqa: E731
        else:
            raise TypeError(
                f"channel_config must be an NS3WifiConfig or None, "
                f"got {type(channel_config).__name__}."
            )

        base_space: spaces.Box = env.observation_space
        obs_shape: tuple = base_space.shape
        obs_dtype = base_space.dtype

        # Build the CentralNode (owns channel + buffer for this agent)
        self._central = CentralNode(
            node_ids=[node_id],
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            config=config,
            channel_factory=channel_factory,
        )

        # Override observation_space with fixed-size Dict
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(config.buffer_size, *obs_shape),
                dtype=obs_dtype,
            ),
            "recv_mask": spaces.MultiBinary(config.buffer_size),
        })

    # -----------------------------------------------------------------------
    # gymnasium.Wrapper overrides
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Reset the wrapped environment and clear all channel / buffer state.

        Returns the initial Dict observation (all zeros, all recv_mask=False)
        because no observation has been transmitted or received yet.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._central.reset()
        self._step_count = 0

        obs_buf, recv_mask = self._central.get_buffer(self._node_id)
        return {"observations": obs_buf, "recv_mask": recv_mask}, info

    def step(
        self, action: Any, packet_size: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Step the underlying environment and run the channel simulation.

        Parameters
        ----------
        action      : Any           Action compatible with the wrapped env.
        packet_size : int | None    Payload bytes to use for the packet
                                    transmitted this step.  None means use
                                    the channel's own default (NS3WifiConfig.
                                    packet_size_bytes for the ns-3 backend,
                                    ignored for GE / Perfect channels).

        Sequence per step
        -----------------
        1. env.step(action)           -> raw_obs, reward, term, trunc, info
        2. central.receive_from(...)  -> transmit through channel
        3. central.flush_and_update() -> flush + buffer.add(obs or None)
        4. step_count  += 1
        5. buffer.get_padded()        -> (obs_array, recv_mask)
        6. Return Dict observation, reward, flags, augmented info.

        The `info` dict is extended with:
          "channel_info"      : dict from get_channel_info() (state, params…)
          "arrived_this_step" : bool, True if a packet arrived at this step.
        """
        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        t = self._step_count

        self._central.receive_from(self._node_id, raw_obs, t, packet_size)
        arrived_map = self._central.flush_and_update(t)

        self._step_count += 1

        obs_buf, recv_mask = self._central.get_buffer(self._node_id)

        info["channel_info"] = self._central.get_channel_info(self._node_id)
        info["arrived_this_step"] = arrived_map[self._node_id] is not None

        return (
            {"observations": obs_buf, "recv_mask": recv_mask},
            reward,
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Current integer step counter (0-indexed; incremented after each step)."""
        return self._step_count

    @property
    def config(self) -> NetworkConfig:
        """The NetworkConfig used to configure this wrapper."""
        return self._config

    @property
    def central_node(self) -> CentralNode:
        """Direct access to the underlying CentralNode (for multi-agent use)."""
        return self._central
