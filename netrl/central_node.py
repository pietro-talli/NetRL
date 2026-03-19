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
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from netrl.channels.comm_channel import CommChannel
from netrl.channels.network_config import NetworkConfig
from netrl.utils.observation_buffer import ObservationBuffer


class CentralNode:
    """
    Aggregator that owns one CommChannel and one ObservationBuffer per node.

    Parameters
    ----------
    node_ids        : List[str]
        Unique string identifiers for each distributed node (agent).
    obs_shape       : tuple | List[tuple]
        Shape of a single observation.  Pass a single ``tuple`` (e.g.
        ``(4,)``) to use the same shape for every node, or a ``List[tuple]``
        (one entry per node) to give each node its own observation shape.
    obs_dtype       : dtype | List[dtype]
        NumPy dtype of observations.  Same broadcast rules as ``obs_shape``:
        a single value is applied to all nodes; a list assigns per-node.
    config          : NetworkConfig
        Channel + buffer configuration shared across all nodes.
        Each node gets a copy with ``seed = config.seed + node_index`` so
        the per-node RNGs are independent.
    channel_factory : Callable[[NetworkConfig], CommChannel]
        Callable that takes a NetworkConfig and returns a CommChannel.
        Swap for PerfectChannel, NS3WifiChannel, or any custom channel
        without changing this class.
    """

    def __init__(
        self,
        node_ids: List[str],
        obs_shape: Union[tuple, List[tuple]],
        obs_dtype: Union[np.dtype, List[np.dtype]],
        config: NetworkConfig,
        channel_factory: Callable[[NetworkConfig], CommChannel],
    ) -> None:
        self._node_ids = list(node_ids)
        self._config = config

        self._channels: Dict[str, CommChannel] = {}
        self._buffers: Dict[str, ObservationBuffer] = {}

        # Normalize to per-node lists so the loop below is uniform.
        obs_shapes: List[tuple] = (
            obs_shape if isinstance(obs_shape, list)
            else [obs_shape] * len(node_ids)
        )
        obs_dtypes: List = (
            obs_dtype if isinstance(obs_dtype, list)
            else [obs_dtype] * len(node_ids)
        )

        for i, nid in enumerate(node_ids):
            node_cfg = copy.replace(config, seed=config.seed + i)
            self._channels[nid] = channel_factory(node_cfg)
            self._buffers[nid] = ObservationBuffer(
                maxlen=config.buffer_size,
                shape=obs_shapes[i],
                dtype=obs_dtypes[i],
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
          - flush its channel to retrieve all packets due at this step,
          - add ALL packets to the buffer with their correct observation times

        When a packet arrives at step S with delay_steps=D, the observation it
        contains is from time step (S - D), so we add it with that step number.

        With GEChannel (fixed delay): typically 0-1 packet per step
        With NS3WifiChannel (variable delay): can be 0-N packets per step due to
        retransmissions and variable latencies; we add all of them.

        The arrived_map records the last packet that arrived for each node
        (for backward compatibility and info reporting).

        Returns
        -------
        Dict[node_id -> obs | None]
            The last observation that arrived for each node this step, or None.
        """
        arrived_map: Dict[str, Optional[np.ndarray]] = {}
        for nid in self._node_ids:
            packets = self._channels[nid].flush(step)

            last_obs = None
            if packets:
                # Add ALL packets to the buffer (important for NS3 which can have multiple)
                for arrival_step, obs in packets:
                    # Calculate which observation step this corresponds to
                    obs_step = arrival_step - self._config.delay_steps
                    self._buffers[nid].add(obs, obs_step)
                    last_obs = obs  # Track the last one for arrived_map

            # Always record a step entry to advance the time window
            if not packets:
                # No packet arrived this step, but mark it so buffer advances
                self._buffers[nid].add(None, step)

            # Ensure buffer is aware of current time
            self._buffers[nid].current_step = max(self._buffers[nid].current_step, step)

            arrived_map[nid] = last_obs

        return arrived_map

    # -----------------------------------------------------------------------
    # Buffer access
    # -----------------------------------------------------------------------

    def get_buffer(self, node_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the padded (obs_array, recv_mask) for `node_id`.

        Shapes:
            obs_array : (buffer_size, ``*obs_shape``)
            recv_mask : (buffer_size,) — dtype bool

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
