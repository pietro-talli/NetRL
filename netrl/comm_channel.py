"""
comm_channel.py
===============
Defines the CommChannel abstract base class and two concrete implementations:

  GEChannel      — Gilbert-Elliott channel backed by the C++ pybind11 extension.
  PerfectChannel — Lossless, zero-delay channel for baselines and unit tests.

The ABC is the extensibility seam for future channel backends (e.g. ns3).
To plug in a new backend, subclass CommChannel and implement the four abstract
methods, then pass `channel_factory=YourChannel` to NetworkedEnv or CentralNode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from netrl.network_config import NetworkConfig


class CommChannel(ABC):
    """
    Abstract interface for a communication channel simulation.

    Contract
    --------
    - transmit(obs, step) is called exactly once per env.step() with the raw
      observation produced by the wrapped environment and the current integer
      step counter.
    - flush(step)  is called exactly once per env.step() (after transmit) and
      returns all packets whose scheduled arrival_step <= step.
    - reset() is called on env.reset(); must clear all pending packets and
      any internal channel state.
    - get_channel_info() returns a diagnostic dict for logging; minimum keys:
      {"state": str, "pending_count": int}.

    Fixed-delay channels (GEChannel) guarantee at most one packet returned by
    flush() per step.  Variable-delay channels may return more.
    """

    @abstractmethod
    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: Optional[int] = None) -> None:
        """
        Simulate transmission of `obs` at integer step `step`.

        The channel decides whether the packet is lost and, if not, computes
        a delivery step (>= step) and queues the packet internally.

        Parameters
        ----------
        obs         : np.ndarray  Raw observation from the wrapped env.
        step        : int         Current integer step counter (0-indexed).
        packet_size : int | None  Payload size in bytes for this packet.
                                  None means use the channel's default.
                                  Channels that do not model packet-size
                                  effects (GE, Perfect) silently ignore it.
        """

    @abstractmethod
    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        """
        Return all packets whose arrival_step <= step.

        Each element is a tuple (arrival_step: int, obs: np.ndarray).
        Returns an empty list when no packet is due.

        Parameters
        ----------
        step : int  Current integer step counter.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Clear pending packets and reset internal state.
        Called on env.reset(). Must NOT re-seed the RNG.
        """

    @abstractmethod
    def get_channel_info(self) -> dict:
        """
        Return a diagnostic dict for logging or debugging.

        Minimum keys: {"state": str, "pending_count": int}.
        """


# ---------------------------------------------------------------------------


class GEChannel(CommChannel):
    """
    Gilbert-Elliott channel backed by the C++ pybind11 extension `netcomm`.

    The Markov chain state (Good/Bad), RNG, and pending packet queue all live
    inside the C++ GEChannelImpl object for atomicity and reproducibility.

    Parameters
    ----------
    config : NetworkConfig
        Channel and buffer configuration.  delay_steps, p_gb, p_bg,
        loss_good, loss_bad, and seed are forwarded to the C++ backend.

    Raises
    ------
    ImportError
        If the netcomm C++ extension has not been built.
        Run `pip install -e .` or `python setup.py build_ext --inplace`.
    """

    def __init__(self, config: NetworkConfig) -> None:
        try:
            import netcomm  # C++ pybind11 extension
        except ImportError as exc:
            raise ImportError(
                "netcomm C++ extension not found. "
                "Run `pip install -e .` or "
                "`python setup.py build_ext --inplace`."
            ) from exc

        self._impl = netcomm.GEChannelImpl(
            p_gb=config.p_gb,
            p_bg=config.p_bg,
            loss_good=config.loss_good,
            loss_bad=config.loss_bad,
            delay_steps=config.delay_steps,
            seed=config.seed,
        )

    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: Optional[int] = None) -> None:
        # packet_size has no effect on the GE channel model; ignored.
        self._impl.transmit(np.ascontiguousarray(obs, dtype=np.float64), step)

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        return self._impl.flush(step)

    def reset(self) -> None:
        self._impl.reset()

    def get_channel_info(self) -> dict:
        return dict(self._impl.get_channel_info())


# ---------------------------------------------------------------------------


class PerfectChannel(CommChannel):
    """
    Lossless, zero-delay channel for debugging and baselines.

    Does not require the C++ extension.  Every transmitted packet is
    immediately available at the same step via flush().
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        # config is accepted but ignored; included for API compatibility
        # with channel_factory(config) call signature.
        self._pending: List[Tuple[int, np.ndarray]] = []

    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: Optional[int] = None) -> None:
        # packet_size has no effect on the perfect channel; ignored.
        self._pending.append((step, obs.copy()))

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        due = [(s, o) for s, o in self._pending if s <= step]
        self._pending = [(s, o) for s, o in self._pending if s > step]
        return due

    def reset(self) -> None:
        self._pending.clear()

    def get_channel_info(self) -> dict:
        return {
            "state": "PERFECT",
            "pending_count": len(self._pending),
        }
