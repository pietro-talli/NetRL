from dataclasses import dataclass, field


@dataclass
class NS3WifiConfig:
    """
    Configuration for the NS3WifiChannel — ns-3 802.11a single-hop WiFi link.

    Physical / channel parameters
    -----------------------------
    These are forwarded as command-line arguments to the ns3_wifi_sim subprocess
    at startup and stay fixed for the lifetime of the channel (until reset).

    distance_m : float
        STA-to-AP Euclidean distance in metres.  Larger distances raise path
        loss and increase both loss probability and MAC retransmission delay.
        For 802.11a at 20 dBm TX, usable range with path-loss exponent 3 is
        roughly 0–80 m depending on the loss model.

    step_duration_ms : float
        Width of one environment step in ns-3 simulation time (milliseconds).
        Each env step = one ns-3 interval.  Smaller values give finer temporal
        resolution of WiFi delay (e.g. 1 ms ≈ 1 MAC frame time at 54 Mbps),
        while larger values (e.g. 10 ms) let many retransmissions complete
        within a single step, giving mostly zero-step delay with occasional
        drops.  Recommended: 1–10 ms.

    tx_power_dbm : float
        Transmit power of the STA in dBm.

    loss_exponent : float
        Path-loss exponent n for the log-distance model:
            PL(d) = PL_ref + 10 * n * log10(d / d_ref)
        Typical values: 2 (free space), 3 (indoor/outdoor mixed),
        4 (dense indoor).

    max_retries : int
        Maximum number of MAC-layer retransmission attempts before a frame
        is declared undeliverable and dropped.  Each retry adds ~3–5 ms of
        delay (DIFS + backoff + ACK timeout).  ns-3 default is 7.

    packet_size_bytes : int
        Total UDP payload in bytes for the probe packet.  The first 4 bytes
        carry the step_id; the remainder is padding.  Larger packets have
        higher collision probability and longer transmission time.

    Timing / Python side
    --------------------
    max_pending_steps : int
        Maximum number of env steps a transmitted packet is allowed to be
        in-flight before it is considered lost on the Python side and removed
        from the pending dictionary.  Must be at least
        ceil(max_retransmission_delay_ms / step_duration_ms).
        Default 200 steps is conservative for step_duration_ms >= 1.

    sim_binary : str
        Path to the compiled ns3_wifi_sim binary.  If empty (default), the
        channel auto-detects it relative to this file's location:
            <project_root>/src/ns3_wifi_sim
    """

    # --- Physical layer ---
    distance_m: float = 10.0
    step_duration_ms: float = 1.0
    tx_power_dbm: float = 20.0
    loss_exponent: float = 3.0
    max_retries: int = 7
    packet_size_bytes: int = 64

    # --- Python-side timeout ---
    max_pending_steps: int = 200
    """Steps after which an unacknowledged transmission is declared lost."""

    # --- Binary location ---
    sim_binary: str = ""
    """
    Absolute path to the ns3_wifi_sim binary.
    If empty, auto-detected as <project_root>/src/ns3_wifi_sim.
    """

    def validate(self) -> None:
        """Raise ValueError for out-of-range parameters."""
        if self.distance_m <= 0:
            raise ValueError(f"distance_m={self.distance_m} must be > 0")
        if self.step_duration_ms <= 0:
            raise ValueError(
                f"step_duration_ms={self.step_duration_ms} must be > 0"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries={self.max_retries} must be >= 0"
            )
        if self.packet_size_bytes < 4:
            raise ValueError(
                f"packet_size_bytes={self.packet_size_bytes} must be >= 4 "
                "(first 4 bytes carry the step_id)"
            )
        if self.max_pending_steps < 1:
            raise ValueError(
                f"max_pending_steps={self.max_pending_steps} must be >= 1"
            )
