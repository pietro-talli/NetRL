"""
ns3_channel.py
==============
NS3WifiChannel — a CommChannel backed by an ns-3 WiFi simulation running
as a persistent subprocess.

Architecture
------------
The channel spawns a single long-lived child process (src/ns3_wifi_sim)
that owns the full ns-3 event loop.  Python and the subprocess communicate
over stdin/stdout using a line-oriented text protocol:

    TRANSMIT <step_id> <size>  →  schedule a probe packet for env step step_id
    FLUSH    <step_id>  →  advance ns-3 to end of step_id, report arrivals
    RESET               →  restart the ns-3 simulation (sim time → 0)
    QUIT                →  clean shutdown

The subprocess responds with:
    READY               →  emitted once at startup
    OK                  →  acknowledges TRANSMIT / RESET
    RECV <id>…          →  space-separated step_ids that arrived (may be empty)
    ERROR <msg>         →  unexpected condition

Timing model
------------
• Each env step t occupies ns-3 time [t·step_ms, (t+1)·step_ms).
• transmit(obs, t) schedules a packet send 1 % into step t's window.
• flush(t)        advances the sim to (t+1)·step_ms and collects arrivals.

The ns-3 simulation is PERSISTENT across env steps.  The full MAC / PHY
state (backoff counters, retry timers, etc.) persists between consecutive
steps, giving temporally-correlated realistic channel behaviour.  The
simulation is only torn down and rebuilt by reset() (i.e. env.reset()).

Observation mapping
-------------------
The step_id embedded in each probe packet (4-byte big-endian uint32)
allows the Python side to map a received packet back to the correct
observation, even when packets arrive out of their transmit order.

A pending observation is expired (considered lost) after max_pending_steps
env steps without receiving an acknowledgement.  This bounds memory usage
and handles the case where ns-3 silently drops a packet (MAC retry
exhaustion never fires a Python callback).

Usage
-----
    from netrl import NetworkedEnv, NetworkConfig, NS3WifiChannel, NS3WifiConfig
    import gymnasium as gym

    ns3_cfg = NS3WifiConfig(distance_m=15.0, step_duration_ms=2.0)
    config  = NetworkConfig(buffer_size=20)

    env = NetworkedEnv(
        gym.make("CartPole-v1"),
        config,
        channel_factory=lambda cfg: NS3WifiChannel(cfg, ns3_cfg),
    )

Build the binary first:
    bash src/build_ns3_sim.sh
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from netrl.channels.comm_channel import CommChannel
from netrl.channels.network_config import NetworkConfig
from netrl.channels.ns3_wifi_config import NS3WifiConfig

# Default location of the compiled ns-3 simulation binary relative to
# this file: <project_root>/src/ns3_wifi_sim
_DEFAULT_SIM_BINARY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../..", "src", "ns3_wifi_sim")
)


class NS3WifiChannel(CommChannel):
    """
    CommChannel implementation backed by an ns-3 802.11a WiFi simulation.

    Each instance manages exactly one subprocess running ns3_wifi_sim.
    The subprocess is started on construction and stays alive until
    reset() rebuilds the simulation or the Python object is garbage-collected.

    Parameters
    ----------
    config     : NetworkConfig   NetRL shared config (buffer_size, seed etc.).
                                 Note: GE-specific fields (p_gb, p_bg, …) are
                                 not used here — WiFi channel physics determine
                                 loss and delay.
    ns3_config : NS3WifiConfig   ns-3-specific physical-layer parameters.
                                 If None, defaults are used.
    """

    def __init__(
        self,
        config: NetworkConfig,
        ns3_config: Optional[NS3WifiConfig] = None,
    ) -> None:
        self._config = config
        self._ns3_cfg: NS3WifiConfig = ns3_config or NS3WifiConfig()
        self._ns3_cfg.validate()

        # step_id → (obs, sent_at_step): obs waiting for the ns-3 ack
        self._pending: Dict[int, Tuple[np.ndarray, int]] = {}

        # Packets confirmed received by ns-3, waiting to be returned by flush()
        # Each item: (arrival_step, obs)
        self._arrived: deque = deque()

        self._proc: Optional[subprocess.Popen] = None
        # Rolling buffer of recent stderr lines (populated by background thread)
        self._stderr_buf: deque = deque(maxlen=200)
        self._start_subprocess()

    # -----------------------------------------------------------------------
    # CommChannel interface
    # -----------------------------------------------------------------------

    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: Optional[int] = None) -> None:
        """
        Instruct ns-3 to simulate sending the observation at env step `step`.

        The probe packet carrying the step_id is scheduled inside the ns-3
        subprocess at simulation time step * step_ms + 0.01 * step_ms.
        The observation is stored locally; ns-3 only carries the 4-byte
        step_id, not the full observation data.

        Parameters
        ----------
        obs         : np.ndarray  Raw observation from the wrapped env.
        step        : int         Current integer step counter (0-indexed).
        packet_size : int | None  Payload size in bytes.  None uses the
                                  default from NS3WifiConfig.packet_size_bytes.
        """
        size = packet_size if packet_size is not None else self._ns3_cfg.packet_size_bytes
        self._pending[step] = (obs.copy(), step)
        self._send_command(f"TRANSMIT {step} {size}")
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(
                f"NS3WifiChannel transmit: unexpected response '{resp}'"
            )

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        """
        Advance the ns-3 simulation to the end of env step `step` and
        return all observations whose packets arrived within this window.

        Observations from earlier steps that had multi-step delays (due to
        WiFi MAC retransmissions) may appear here.

        Additionally, pending observations older than max_pending_steps are
        expired (considered permanently lost).

        Parameters
        ----------
        step : int  Current integer step counter.

        Returns
        -------
        List of (arrival_step, obs) tuples.  arrival_step is set to `step`
        (the flush step) regardless of the exact ns-3 arrival time within
        the window, because the CommChannel contract only requires that
        return values have arrival_step <= step.
        """
        # Ask the ns-3 subprocess to run until end of this step
        self._send_command(f"FLUSH {step}")
        response = self._read_line()

        if not response.startswith("RECV"):
            raise RuntimeError(
                f"NS3WifiChannel flush: unexpected response '{response}'"
            )

        # Parse arrived step_ids
        parts = response.split()
        for part in parts[1:]:
            sid = int(part)
            if sid in self._pending:
                obs, _ = self._pending.pop(sid)
                self._arrived.append((step, obs))

        # Expire old pending observations (packet permanently lost / dropped)
        expired = [
            sid
            for sid, (_, sent_step) in self._pending.items()
            if step - sent_step > self._ns3_cfg.max_pending_steps
        ]
        for sid in expired:
            self._pending.pop(sid)

        # Drain all arrivals whose arrival_step <= step into result
        result: List[Tuple[int, np.ndarray]] = []
        while self._arrived and self._arrived[0][0] <= step:
            result.append(self._arrived.popleft())

        return result

    def reset(self) -> None:
        """
        Restart the ns-3 simulation (simulation time → 0) and clear all
        pending / arrived state.  Called on env.reset().

        The subprocess stays alive; only the ns-3 internal simulator state
        is destroyed and rebuilt.  This is equivalent to starting a new
        wireless scenario.
        """
        self._pending.clear()
        self._arrived.clear()
        self._send_command("RESET")
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(
                f"NS3WifiChannel reset: unexpected response '{resp}'"
            )

    def get_channel_info(self) -> dict:
        """
        Return diagnostic information about the channel state.

        Does not query the subprocess (would require an extra round-trip);
        derives info from Python-side bookkeeping instead.
        """
        return {
            "state":          "NS3_WIFI",
            "pending_count":  len(self._pending),
            "arrived_buffered": len(self._arrived),
            "distance_m":       self._ns3_cfg.distance_m,
            "step_duration_ms": self._ns3_cfg.step_duration_ms,
            "tx_power_dbm":     self._ns3_cfg.tx_power_dbm,
            "loss_exponent":    self._ns3_cfg.loss_exponent,
            "max_retries":      self._ns3_cfg.max_retries,
        }

    # -----------------------------------------------------------------------
    # Subprocess management
    # -----------------------------------------------------------------------

    def _resolve_binary(self) -> str:
        """Return the path to the ns3_wifi_sim binary, raising if not found."""
        path = self._ns3_cfg.sim_binary or _DEFAULT_SIM_BINARY
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ns3_wifi_sim binary not found at '{path}'.\n"
                "Build it first:\n"
                "    bash src/build_ns3_sim.sh\n"
                "Or set NS3WifiConfig.sim_binary to the correct path."
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"ns3_wifi_sim binary at '{path}' is not executable.\n"
                "Run: chmod +x " + path
            )
        return path

    def _start_subprocess(self) -> None:
        """Launch the ns3_wifi_sim subprocess and wait for READY."""
        binary = self._resolve_binary()
        cfg    = self._ns3_cfg

        cmd = [
            binary,
            f"--step-ms={cfg.step_duration_ms}",
            f"--distance={cfg.distance_m}",
            f"--tx-power={cfg.tx_power_dbm}",
            f"--loss-exp={cfg.loss_exponent}",
            f"--retries={cfg.max_retries}",
            f"--pkt-size={cfg.packet_size_bytes}",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # ns-3 log/warnings → captured stderr
            bufsize=0,                # unbuffered byte-stream
            text=True,
            encoding="utf-8",
        )

        # Background thread: continuously drain stderr so the pipe never fills
        # up and blocks the subprocess.  Lines are stored in _stderr_buf for
        # error reporting.
        def _drain_stderr_loop() -> None:
            try:
                for line in self._proc.stderr:  # type: ignore[union-attr]
                    self._stderr_buf.append(line.rstrip("\n\r"))
            except Exception:
                pass

        threading.Thread(target=_drain_stderr_loop, daemon=True).start()

        # Wait for the READY signal (with timeout)
        try:
            ready_line = self._read_line(timeout=30.0)
        except TimeoutError:
            self._kill_subprocess()
            raise RuntimeError(
                "ns3_wifi_sim did not emit READY within 30 s. "
                "Check stderr for ns-3 error messages."
            )

        if ready_line != "READY":
            stderr_preview = self._drain_stderr()
            self._kill_subprocess()
            raise RuntimeError(
                f"ns3_wifi_sim emitted unexpected startup line: '{ready_line}'\n"
                f"stderr: {stderr_preview}"
            )

    def _kill_subprocess(self) -> None:
        """Terminate the subprocess forcefully."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _send_command(self, line: str) -> None:
        """Write a command line to the subprocess stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("NS3WifiChannel: subprocess is not running.")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                f"NS3WifiChannel: subprocess stdin pipe broken.\n"
                f"stderr: {stderr_preview}"
            ) from exc

    def _read_line(self, timeout: float = 10.0) -> str:
        """Read one response line from the subprocess stdout (strips \\n)."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("NS3WifiChannel: subprocess is not running.")

        # Use a thread to implement the read timeout
        result: List[Optional[str]] = [None]
        exc_holder: List[Optional[Exception]] = [None]

        def _read():
            try:
                result[0] = self._proc.stdout.readline()  # type: ignore[union-attr]
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            stderr_preview = self._drain_stderr()
            raise TimeoutError(
                f"NS3WifiChannel: subprocess did not respond within {timeout}s.\n"
                f"stderr: {stderr_preview}"
            )
        if exc_holder[0] is not None:
            raise exc_holder[0]

        line = result[0]
        if line is None or line == "":
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                "NS3WifiChannel: subprocess stdout closed (process exited?).\n"
                f"Return code: {self._proc.poll()}\n"
                f"stderr: {stderr_preview}"
            )
        return line.rstrip("\n\r")

    def _drain_stderr(self, max_lines: int = 50) -> str:
        """Return recent stderr lines from the background drain buffer."""
        if not self._stderr_buf:
            return "<no stderr>"
        lines = list(self._stderr_buf)[-max_lines:]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def __del__(self) -> None:
        """Send QUIT before the object is garbage-collected."""
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send_command("QUIT")
                self._proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            self._kill_subprocess()
