from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NS3WifiMultiUEConfig:
    """
    Configuration for a multi-UE ns-3 802.11a infrastructure WiFi network.

    Topology: N UEs (STAs) all associated with a single AP (central node).
    All UEs share the same 802.11a channel and contend via CSMA/CA, producing
    realistic multi-node uplink behaviour.

    Parameters
    ----------
    n_ues : int
        Number of UE nodes (STAs).  Must match len(node_ids) passed to
        CentralNode and the factory returned by make_multi_ue_wifi_factory().

    distances_m : List[float]
        Euclidean distance (metres) from each UE to the AP.  Element i
        is the distance for UE i.  If fewer values than n_ues are given,
        the last value is repeated for the remaining UEs.

    step_duration_ms : float
        Width of one environment step in ns-3 simulation time (ms).
        See NS3WifiConfig for guidance on choosing this value.

    tx_power_dbm : float
        Transmit power of every STA in dBm.

    loss_exponent : float
        Path-loss exponent for the log-distance model (2 = free space,
        3 = mixed indoor/outdoor, 4 = dense indoor).

    max_retries : int
        Maximum MAC-layer retransmission attempts per frame.

    packet_size_bytes : int
        UDP payload in bytes.  The first 8 bytes carry the (ue_id, step_id)
        header; the rest is padding.  Must be >= 8.

    max_pending_steps : int
        A transmitted packet not acknowledged after this many env steps is
        considered lost on the Python side and purged from memory.

    sim_binary : str
        Path to the compiled ns3_wifi_multi_ue_sim binary.
        If empty, auto-detected as <project_root>/src/ns3_wifi_multi_ue_sim.
    """

    # --- Network topology ---
    n_ues: int = 2
    distances_m: List[float] = field(default_factory=lambda: [10.0, 10.0])

    # --- Physical / channel parameters ---
    step_duration_ms: float = 1.0
    tx_power_dbm: float = 20.0
    loss_exponent: float = 3.0
    max_retries: int = 7
    packet_size_bytes: int = 64

    # --- Python-side timeout ---
    max_pending_steps: int = 200

    # --- Binary location ---
    sim_binary: str = ""

    def validate(self) -> None:
        """Raise ValueError for out-of-range or inconsistent parameters."""
        if self.n_ues < 1:
            raise ValueError(f"n_ues={self.n_ues} must be >= 1")
        if not self.distances_m:
            raise ValueError("distances_m must not be empty")
        for i, d in enumerate(self.distances_m):
            if d <= 0:
                raise ValueError(f"distances_m[{i}]={d} must be > 0")
        if self.step_duration_ms <= 0:
            raise ValueError(
                f"step_duration_ms={self.step_duration_ms} must be > 0"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries={self.max_retries} must be >= 0"
            )
        if self.packet_size_bytes < 8:
            raise ValueError(
                f"packet_size_bytes={self.packet_size_bytes} must be >= 8 "
                "(first 8 bytes carry the ue_id + step_id header)"
            )
        if self.max_pending_steps < 1:
            raise ValueError(
                f"max_pending_steps={self.max_pending_steps} must be >= 1"
            )
