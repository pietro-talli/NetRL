"""
networked_env.py
================
NetworkedEnv — a gymnasium.Wrapper that simulates a noisy communication
channel between the RL agent and the environment.

At every step the wrapper:
  1. Steps the underlying environment to get the raw observation.
  2. Calls the C++ Gilbert-Elliott subroutine (via CentralNode) to
     simulate transmission to the central node (applies loss + delay).
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

Plugging in a different channel backend
---------------------------------------
Pass a custom `channel_factory` callable:

    env = NetworkedEnv(
        gym.make("CartPole-v1"), config,
        channel_factory=PerfectChannel,     # or NS3Channel, etc.
    )

No other change is needed.  The factory must accept a NetworkConfig and
return a CommChannel instance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from netrl.central_node import CentralNode
from netrl.comm_channel import CommChannel, GEChannel
from netrl.network_config import NetworkConfig


class NetworkedEnv(gym.Wrapper):
    """
    Gymnasium wrapper simulating networked observation transmission.

    Parameters
    ----------
    env             : gymnasium.Env
        The base environment to wrap.  Must have a Box observation space.
    config          : NetworkConfig
        Channel and buffer configuration.  Validated on construction.
    channel_factory : Callable[[NetworkConfig], CommChannel]
        Factory that builds the channel backend.  Defaults to GEChannel
        (C++ Gilbert-Elliott).  Pass PerfectChannel for a baseline or
        a custom NS3Channel for ns3-backed simulation.
    node_id         : str
        Identifier for this agent's transmission node.  Default "agent_0".
        Change only if you need a non-default key in multi-agent setups.
    """

    def __init__(
        self,
        env: gym.Env,
        config: NetworkConfig,
        channel_factory: Callable[[NetworkConfig], CommChannel] = GEChannel,
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
        self, action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Step the underlying environment and run the channel simulation.

        Sequence per step
        -----------------
        1. env.step(action)           -> raw_obs, reward, term, trunc, info
        2. central.receive_from(...)  -> C++ transmit (loss + delay)
        3. central.flush_and_update() -> C++ flush + buffer.add(obs or None)
        4. step_count  += 1
        5. buffer.get_padded()        -> (obs_array, recv_mask)
        6. Return Dict observation, reward, flags, augmented info.

        The `info` dict is extended with:
          "channel_info"      : dict from get_channel_info() (state, params…)
          "arrived_this_step" : bool, True if a packet arrived at this step.

        Parameters
        ----------
        action : Any  Action compatible with the wrapped env's action_space.

        Returns
        -------
        obs        : Dict{"observations": ndarray, "recv_mask": ndarray}
        reward     : float
        terminated : bool
        truncated  : bool
        info       : dict
        """
        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        t = self._step_count

        # Transmit raw observation through the channel (C++ subroutine)
        self._central.receive_from(self._node_id, raw_obs, t)

        # Flush channel: collect any packet(s) due at step t, update buffer
        arrived_map = self._central.flush_and_update(t)

        self._step_count += 1

        # Build Dict observation from padded buffer
        obs_buf, recv_mask = self._central.get_buffer(self._node_id)

        # Augment info with channel diagnostics
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
