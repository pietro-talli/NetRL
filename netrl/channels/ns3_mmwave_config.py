from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NS3MmWaveConfig:
    """
    Configuration for NS3MmWaveChannel — ns-3 5G mmWave single-hop link.

    All parameters are forwarded as command-line arguments to the
    ``ns3_mmwave_sim`` subprocess.  Build it once with::

        bash src/build_ns3_mmwave_sim.sh

    Physical-layer parameters
    -------------------------
    distance_m : float
        UE-to-eNB Euclidean distance in metres.  At 28 GHz with path-loss
        exponent ~3.5 (Urban Macro), usable range before complete link
        failure is roughly 0–200 m.  The model uses height-dependent
        LOS/NLOS probability from the configured 3GPP scenario.

    frequency_ghz : float
        Centre carrier frequency in GHz.  Common 5G mmWave bands:

        +--------+------------------+----------+
        |  Band  | Frequency        | Range    |
        +--------+------------------+----------+
        | n257   | 26.5 – 29.5 GHz  | ~150 m   |
        | n258   | 24.25 – 27.5 GHz | ~200 m   |
        | n260   | 37 – 40 GHz      | ~100 m   |
        | n261   | 27.5 – 28.35 GHz | ~150 m   |
        +--------+------------------+----------+

    bandwidth_ghz : float
        Component carrier bandwidth in GHz.  Typical NR mmWave values:

        +------------------+-------------------------------------------+
        | Bandwidth (GHz)  | Peak PHY throughput @ 28 GHz, 64-QAM     |
        +------------------+-------------------------------------------+
        | 0.05   (50 MHz)  | ~350 Mbps                                 |
        | 0.1   (100 MHz)  | ~700 Mbps                                 |
        | 0.2   (200 MHz)  | ~ 1.4 Gbps                                |
        | 0.4   (400 MHz)  | ~ 2.8 Gbps                                |
        | 0.8   (800 MHz)  | ~ 5.6 Gbps                                |
        +------------------+-------------------------------------------+

    tx_power_dbm : float
        UE transmit power in dBm.  Typical 5G mmWave UE: 23 dBm.
        Higher power improves uplink SNR and reduces retransmissions.

    enb_tx_power_dbm : float
        eNB (gNB) transmit power in dBm.  Typical: 30 dBm.
        Affects downlink quality (not directly observed here since we
        measure uplink), but influences SINR feedback and scheduling.

    noise_figure_db : float
        UE receiver noise figure in dB.  Typical mmWave UE: 7–10 dB.
        Higher values degrade received SINR and increase packet errors.

    enb_noise_figure_db : float
        eNB receiver noise figure in dB.  Typical mmWave base station: 5 dB.

    scenario : str
        3GPP TR 38.901 propagation scenario.  Determines the LOS/NLOS
        probability model and path-loss formula:

        ``"RMa"``              Rural Macro — low density, long range
        ``"UMa"``              Urban Macro — default, typical outdoor city
        ``"UMi-StreetCanyon"`` Urban Micro — dense urban, street level
        ``"InH-OfficeMixed"``  Indoor Hotspot — mixed LOS/NLOS office
        ``"InH-OfficeOpen"``   Indoor Hotspot — open-plan office

    harq_enabled : bool
        Enable Hybrid ARQ (HARQ) with incremental redundancy.  When True,
        retransmissions combine with previous attempts to improve decoding
        probability.  Adds 1–5 ms per retransmission round.

    rlc_am_enabled : bool
        Enable RLC Acknowledged Mode (AM).  Adds RLC-level retransmissions
        on top of HARQ, increasing reliability at the cost of additional
        delay.  Recommended False for delay-sensitive observations.

    Timing / Python-side parameters
    --------------------------------
    step_duration_ms : float
        Width of one environment step in ns-3 simulation time (ms).
        Each env step = one ns-3 time window.

        +------------------+-----------------------------------------------+
        | step_duration_ms | Behaviour                                     |
        +------------------+-----------------------------------------------+
        | 0.5 – 1 ms       | ~1 TTI; single HARQ round per step;           |
        |                  | realistic per-packet delay variation          |
        | 2 – 5 ms         | 2–5 HARQ rounds fit per step; most packets    |
        |                  | arrive same step or are dropped               |
        | 10+ ms           | Very coarse; near-zero delay variation        |
        +------------------+-----------------------------------------------+

    packet_size_bytes : int
        Default UDP payload in bytes for the probe packet.  The first 4
        bytes carry the step_id; the remainder is padding.  Larger packets
        have longer transmission time (especially at lower MCS) and higher
        block error probability.

    max_pending_steps : int
        Maximum env steps a transmitted packet may be in-flight before it
        is declared lost on the Python side.  The mmWave MAC + HARQ can
        take up to ~10 ms to complete one transmission attempt, and with
        ``rlc_am_enabled=True`` additional RLC retransmissions are possible.
        Default 500 is conservative for step_duration_ms >= 1.

    sim_binary : str
        Absolute path to the ``ns3_mmwave_sim`` binary.  Auto-detected as
        ``<project_root>/src/ns3_mmwave_sim`` when empty.
    """

    # --- Radio / physical layer ---
    distance_m:          float = 50.0
    frequency_ghz:       float = 28.0      # GHz  →  passed as Hz to ns-3
    bandwidth_ghz:       float = 0.2       # GHz  →  passed as Hz to ns-3
    tx_power_dbm:        float = 23.0      # UE TX power
    enb_tx_power_dbm:    float = 30.0      # eNB TX power
    noise_figure_db:     float = 9.0       # UE noise figure
    enb_noise_figure_db: float = 5.0       # eNB noise figure
    scenario:            str   = "UMa"     # 3GPP TR 38.901 scenario

    # --- Protocol stack ---
    harq_enabled:   bool = True
    rlc_am_enabled: bool = False

    # --- Packet ---
    packet_size_bytes: int = 64

    # --- Python-side ---
    step_duration_ms:   float = 1.0
    max_pending_steps:  int   = 500
    sim_binary:         str   = ""

    def validate(self) -> None:
        """Raise ``ValueError`` if any parameter is outside its valid range."""
        if self.distance_m <= 0:
            raise ValueError(f"distance_m={self.distance_m!r} must be > 0")
        if self.frequency_ghz <= 0:
            raise ValueError(f"frequency_ghz={self.frequency_ghz!r} must be > 0")
        if self.bandwidth_ghz <= 0:
            raise ValueError(f"bandwidth_ghz={self.bandwidth_ghz!r} must be > 0")
        if self.step_duration_ms <= 0:
            raise ValueError(f"step_duration_ms={self.step_duration_ms!r} must be > 0")
        if self.packet_size_bytes < 4:
            raise ValueError(
                f"packet_size_bytes={self.packet_size_bytes!r} must be >= 4 "
                "(first 4 bytes carry the step_id)"
            )
        if self.max_pending_steps < 1:
            raise ValueError(f"max_pending_steps={self.max_pending_steps!r} must be >= 1")
        valid_scenarios = {
            "RMa", "UMa", "UMi-StreetCanyon", "InH-OfficeMixed", "InH-OfficeOpen"
        }
        if self.scenario not in valid_scenarios:
            raise ValueError(
                f"scenario={self.scenario!r} is not a recognised 3GPP scenario. "
                f"Choose from: {sorted(valid_scenarios)}"
            )
