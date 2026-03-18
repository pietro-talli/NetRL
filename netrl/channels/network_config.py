from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """
    All parameters required to instantiate a Gilbert-Elliott channel
    and an ObservationBuffer.

    Gilbert-Elliott two-state Markov chain
    ----------------------------------------
    States: GOOD (0), BAD (1).

    Transition matrix:
        GOOD -> BAD  with probability p_gb  per step
        BAD  -> GOOD with probability p_bg  per step
        (self-loops: 1 - p_gb and 1 - p_bg respectively)

    Steady-state probability of BAD = p_gb / (p_gb + p_bg).

    In state GOOD, each packet is LOST with probability loss_good.
    In state BAD,  each packet is LOST with probability loss_bad.

    Successfully transmitted packets arrive after exactly delay_steps steps.
    """

    # --- Gilbert-Elliott channel ---
    p_gb: float = 0.1
    """Probability of transitioning Good -> Bad per step."""

    p_bg: float = 0.3
    """Probability of transitioning Bad -> Good per step."""

    loss_good: float = 0.01
    """Packet loss probability in the Good state."""

    loss_bad: float = 0.20
    """Packet loss probability in the Bad state."""

    delay_steps: int = 0
    """Fixed one-way propagation delay expressed in environment steps."""

    # --- Observation buffer ---
    buffer_size: int = 10
    """Number of time slots retained in the ObservationBuffer (window length)."""

    # --- Reproducibility ---
    seed: int = 42
    """RNG seed forwarded to the C++ Gilbert-Elliott backend."""

    def validate(self) -> None:
        """Raise ValueError if any parameter is outside its valid range."""
        for name, val in [
            ("p_gb", self.p_gb),
            ("p_bg", self.p_bg),
            ("loss_good", self.loss_good),
            ("loss_bad", self.loss_bad),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name}={val} must be in [0.0, 1.0]")
        if self.delay_steps < 0:
            raise ValueError(
                f"delay_steps={self.delay_steps} must be >= 0"
            )
        if self.buffer_size < 1:
            raise ValueError(
                f"buffer_size={self.buffer_size} must be >= 1"
            )
