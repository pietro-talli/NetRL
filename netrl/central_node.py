"""
central_node.py
===============
CentralNode manages one CommChannel + ObservationBuffer per registered
distributed node.

For single-agent use NetworkedEnv registers a single node_id internally.
For multi-agent scenarios the caller may register multiple node_ids;
each gets an independent channel instance (same config, seed offset by
node index) and an independent observation buffer.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from netrl.comm_channel import CommChannel
from netrl.network_config import NetworkConfig
from netrl.observation_buffer import ObservationBuffer


class CentralNode:
    """
    Aggregator that owns one CommChannel and one ObservationBuffer per node.

    Parameters
    ----------
    node_ids        : List[str]
        Unique string identifiers for each distributed node (agent).
    obs_shape       : tuple
        Shape of a single observation (e.g. (4,) for CartPole).
    obs_dtype
        NumPy dtype of observations (e.g. np.float32).
    config          : NetworkConfig
        Channel + buffer configuration shared across all nodes.
        Each node gets a copy with seed = config.seed + node_index so
        the per-node RNGs are independent.
    channel_factory : Callable[[NetworkConfig], CommChannel]
        Callable that takes a NetworkConfig and returns a CommChannel.
        Defaults to GEChannel. Swap for PerfectChannel or NS3Channel without
        changing this class.
    """

    def __init__(
        self,
        node_ids: List[str],
        obs_shape: tuple,
        obs_dtype,
        config: NetworkConfig,
        channel_factory: Callable[[NetworkConfig], CommChannel],
    ) -> None:
        self._node_ids = list(node_ids)
        self._config = config

        self._channels: Dict[str, CommChannel] = {}
        self._buffers: Dict[str, ObservationBuffer] = {}

        for i, nid in enumerate(node_ids):
            node_cfg = copy.replace(config, seed=config.seed + i)
            self._channels[nid] = channel_factory(node_cfg)
            self._buffers[nid] = ObservationBuffer(
                maxlen=config.buffer_size,
                shape=obs_shape,
                dtype=obs_dtype,
            )

    # -----------------------------------------------------------------------
    # Per-step API (called by NetworkedEnv.step())
    # -----------------------------------------------------------------------

    def receive_from(self, node_id: str, obs: np.ndarray, step: int,
                     packet_size: Optional[int] = None) -> None:
        """
        Transmit `obs` from `node_id` through its channel.

        Parameters
        ----------
        node_id     : str          Must match one of the ids given at construction.
        obs         : np.ndarray   Raw local observation to be transmitted.
        step        : int          Current integer step counter.
        packet_size : int | None   Payload bytes for this packet.  None means
                                   use the channel's own default.
        """
        if node_id not in self._channels:
            raise KeyError(f"Unknown node_id '{node_id}'")
        self._channels[node_id].transmit(obs, step, packet_size)

    def flush_and_update(self, step: int) -> Dict[str, Optional[np.ndarray]]:
        """
        Flush all channels for `step` and update each observation buffer.

        For each node:
          - flush its channel to retrieve any packet due at this step,
          - call buffer.add(obs) if a packet arrived, else buffer.add(None).

        Invariant: exactly one buffer.add() call per node per step.

        Returns
        -------
        Dict[node_id -> obs | None]
            The observation that arrived for each node this step, or None.
        """
        arrived_map: Dict[str, Optional[np.ndarray]] = {}
        for nid in self._node_ids:
            packets = self._channels[nid].flush(step)
            if packets:
                # With fixed delay, at most one packet arrives per step.
                # If more arrive (variable delay / future ns3), take the last.
                _, obs = packets[-1]
                self._buffers[nid].add(obs)
                arrived_map[nid] = obs
            else:
                self._buffers[nid].add(None)
                arrived_map[nid] = None
        return arrived_map

    # -----------------------------------------------------------------------
    # Buffer access
    # -----------------------------------------------------------------------

    def get_buffer(self, node_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the padded (obs_array, recv_mask) for `node_id`.

        Shapes:
            obs_array : (buffer_size, *obs_shape)
            recv_mask : (buffer_size,)  dtype bool

        The most recent entry is at index [-1]; older entries are to the left.
        Unwritten slots are zeros with recv_mask = False.
        """
        return self._buffers[node_id].get_padded()

    def get_all_buffers(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return get_buffer() for every registered node."""
        return {nid: self.get_buffer(nid) for nid in self._node_ids}

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_channel_info(self, node_id: str) -> dict:
        """Return diagnostic channel state dict for `node_id`."""
        return self._channels[node_id].get_channel_info()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all channels and buffers. Call on env.reset()."""
        for nid in self._node_ids:
            self._channels[nid].reset()
            self._buffers[nid].clear()

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def node_ids(self) -> List[str]:
        return list(self._node_ids)

    @property
    def config(self) -> NetworkConfig:
        return self._config
