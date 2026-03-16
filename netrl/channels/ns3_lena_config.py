from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NS3LenaConfig:
    """
    Configuration for NS3LenaChannel — ns-3 5G-LENA single-cell NR link.

    Parameters are forwarded to the ``ns3_lena_sim`` subprocess.
    Build it once with::

        bash src/build_ns3_lena_sim.sh
    """

    distance_m: float = 50.0
    frequency_ghz: float = 28.0
    bandwidth_ghz: float = 0.1
    ue_tx_power_dbm: float = 23.0
    gnb_tx_power_dbm: float = 30.0
    scenario: str = "UMa"
    numerology: int = 3
    shadowing_enabled: bool = False

    packet_size_bytes: int = 64

    step_duration_ms: float = 1.0
    max_pending_steps: int = 500
    sim_binary: str = ""

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
        if self.numerology < 0 or self.numerology > 5:
            raise ValueError(f"numerology={self.numerology!r} must be in [0, 5]")

        valid_scenarios = {
            "RMa", "UMa", "UMi-StreetCanyon", "InH-OfficeMixed", "InH-OfficeOpen"
        }
        if self.scenario not in valid_scenarios:
            raise ValueError(
                f"scenario={self.scenario!r} is not a recognised 3GPP scenario. "
                f"Choose from: {sorted(valid_scenarios)}"
            )
