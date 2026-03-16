"""
ns3_mmwave_channel.py
=====================
NS3MmWaveChannel — a CommChannel backed by a 5G mmWave ns-3 simulation
running as a persistent subprocess.

Architecture
------------
The channel spawns a single long-lived child process (src/ns3_mmwave_sim)
that owns the full ns-3 event loop with a 5G EPC topology:

    UE ── mmWave PHY/MAC ── eNB ── P2P 100Gbps ── PGW ── Remote Host

Python and the subprocess communicate over stdin/stdout using the same
line-oriented text protocol as NS3WifiChannel:

    TRANSMIT <step_id> <size>  →  schedule a UDP probe packet for env step
    FLUSH    <step_id>  →  advance ns-3 to end of step_id, report arrivals
    RESET               →  destroy & rebuild the ns-3 simulation
    QUIT                →  graceful exit

The subprocess responds with:
    READY               →  emitted once at startup (after 500 ms warm-up)
    OK                  →  acknowledges TRANSMIT / RESET
    RECV <id>…          →  space-separated step_ids that arrived (may be empty)
    ERROR <msg>         →  unexpected condition

Timing model
------------
• Each env step t occupies ns-3 time [t·step_ms, (t+1)·step_ms).
• transmit(obs, t) schedules a packet send 1 % into step t's window.
• flush(t)        advances the sim to (t+1)·step_ms and collects arrivals.

The ns-3 simulation is PERSISTENT across env steps.  The full MAC/PHY
state and channel matrix persist between consecutive steps, giving
temporally-correlated realistic 5G channel behaviour.

Warm-up
-------
The 5G EPC setup requires a warm-up period for:
  - SIB broadcast (FirstSibTime = 2 ms)
  - UE attachment (UseIdealRrc → < 50 ms)
  - Default bearer establishment
The subprocess runs 500 ms of simulation time before emitting READY.
The default READY timeout is therefore 60 s (conservative for slow machines
compiling/loading ns-3 shared libraries).

Usage
-----
    from netrl import NetworkedEnv, NetworkConfig, NS3MmWaveConfig
    import gymnasium as gym

    ns3_cfg = NS3MmWaveConfig(
        distance_m=30.0,
        frequency_ghz=28.0,
        bandwidth_ghz=0.4,
        scenario="UMa",
        step_duration_ms=1.0,
    )
    config  = NetworkConfig(buffer_size=20)

    env = NetworkedEnv(
        gym.make("CartPole-v1"),
        config,
        channel_config=ns3_cfg,
    )

Build the binary first:
    bash src/build_ns3_mmwave_sim.sh
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
from netrl.channels.ns3_mmwave_config import NS3MmWaveConfig

# Default location of the compiled ns-3 mmWave simulation binary relative to
# this file: <project_root>/src/ns3_mmwave_sim
_DEFAULT_SIM_BINARY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../..", "src", "ns3_mmwave_sim")
)


class NS3MmWaveChannel(CommChannel):
    """
    CommChannel implementation backed by an ns-3 5G mmWave EPC simulation.

    Each instance manages exactly one subprocess running ns3_mmwave_sim.
    The subprocess is started on construction, performs a 500 ms warm-up
    (UE attachment + bearer establishment), then emits READY.  It stays
    alive until reset() rebuilds the simulation or the Python object is
    garbage-collected.

    Parameters
    ----------
    config     : NetworkConfig     NetRL shared config (buffer_size, seed …).
                                   Note: GE-specific fields are ignored here —
                                   5G mmWave channel physics determine loss
                                   and delay.
    ns3_config : NS3MmWaveConfig   5G-specific physical-layer parameters.
                                   If None, defaults are used.
    """

    def __init__(
        self,
        config: NetworkConfig,
        ns3_config: Optional[NS3MmWaveConfig] = None,
    ) -> None:
        self._config = config
        self._ns3_cfg: NS3MmWaveConfig = ns3_config or NS3MmWaveConfig()
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

        Parameters
        ----------
        obs         : np.ndarray  Raw observation from the wrapped env.
        step        : int         Current integer step counter (0-indexed).
        packet_size : int | None  Payload size in bytes.  None uses the
                                  default from NS3MmWaveConfig.packet_size_bytes.
        """
        size = packet_size if packet_size is not None else self._ns3_cfg.packet_size_bytes
        self._pending[step] = (obs.copy(), step)
        self._send_command(f"TRANSMIT {step} {size}")
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(
                f"NS3MmWaveChannel transmit: unexpected response '{resp}'"
            )

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        """
        Advance the ns-3 simulation to the end of env step `step` and
        return all observations whose packets arrived within this window.

        Packets from earlier steps that experienced multi-step delays (HARQ
        retransmissions, RLC AM retransmissions) may appear here.

        Pending observations older than max_pending_steps are expired
        (considered permanently lost — MAC exhausted all retransmit attempts).

        Parameters
        ----------
        step : int  Current integer step counter.

        Returns
        -------
        List of (arrival_step, obs) tuples.
        """
        self._send_command(f"FLUSH {step}")
        response = self._read_line()

        if not response.startswith("RECV"):
            raise RuntimeError(
                f"NS3MmWaveChannel flush: unexpected response '{response}'"
            )

        # Parse arrived step_ids
        parts = response.split()
        for part in parts[1:]:
            sid = int(part)
            if sid in self._pending:
                obs, _ = self._pending.pop(sid)
                self._arrived.append((step, obs))

        # Expire old pending observations (packet permanently lost)
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

        The subprocess stays alive; the ns-3 internal state is destroyed
        and rebuilt (new warm-up of 500 ms simulation time).
        """
        self._pending.clear()
        self._arrived.clear()
        self._send_command("RESET")
        resp = self._read_line(timeout=60.0)
        if resp != "OK":
            raise RuntimeError(
                f"NS3MmWaveChannel reset: unexpected response '{resp}'"
            )

    def get_channel_info(self) -> dict:
        """
        Return diagnostic information about the channel state.
        """
        return {
            "state":              "NS3_MMWAVE",
            "pending_count":      len(self._pending),
            "arrived_buffered":   len(self._arrived),
            "distance_m":         self._ns3_cfg.distance_m,
            "frequency_ghz":      self._ns3_cfg.frequency_ghz,
            "bandwidth_ghz":      self._ns3_cfg.bandwidth_ghz,
            "tx_power_dbm":       self._ns3_cfg.tx_power_dbm,
            "enb_tx_power_dbm":   self._ns3_cfg.enb_tx_power_dbm,
            "noise_figure_db":    self._ns3_cfg.noise_figure_db,
            "scenario":           self._ns3_cfg.scenario,
            "harq_enabled":       self._ns3_cfg.harq_enabled,
            "rlc_am_enabled":     self._ns3_cfg.rlc_am_enabled,
            "step_duration_ms":   self._ns3_cfg.step_duration_ms,
        }

    # -----------------------------------------------------------------------
    # Subprocess management
    # -----------------------------------------------------------------------

    def _resolve_binary(self) -> str:
        """Return the path to the ns3_mmwave_sim binary, raising if not found."""
        path = self._ns3_cfg.sim_binary or _DEFAULT_SIM_BINARY
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ns3_mmwave_sim binary not found at '{path}'.\n"
                "Build it first:\n"
                "    bash src/build_ns3_mmwave_sim.sh\n"
                "Or set NS3MmWaveConfig.sim_binary to the correct path."
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"ns3_mmwave_sim binary at '{path}' is not executable.\n"
                "Run: chmod +x " + path
            )
        return path

    def _start_subprocess(self) -> None:
        """Launch the ns3_mmwave_sim subprocess and wait for READY."""
        binary = self._resolve_binary()
        cfg    = self._ns3_cfg

        # frequency_ghz and bandwidth_ghz are stored in GHz;
        # the C++ variables (g_freqHz, g_bandwidthHz) store Hz.
        cmd = [
            binary,
            f"--step-ms={cfg.step_duration_ms}",
            f"--distance={cfg.distance_m}",
            f"--freq={cfg.frequency_ghz * 1e9}",
            f"--bandwidth={cfg.bandwidth_ghz * 1e9}",
            f"--tx-power={cfg.tx_power_dbm}",
            f"--enb-tx-power={cfg.enb_tx_power_dbm}",
            f"--noise-fig={cfg.noise_figure_db}",
            f"--enb-noise-fig={cfg.enb_noise_figure_db}",
            f"--scenario={cfg.scenario}",
            f"--harq={int(cfg.harq_enabled)}",
            f"--rlc-am={int(cfg.rlc_am_enabled)}",
            f"--pkt-size={cfg.packet_size_bytes}",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=True,
            encoding="utf-8",
        )

        # Background thread: continuously drain stderr so the pipe never fills
        # up and blocks the subprocess.
        def _drain_stderr_loop() -> None:
            try:
                for line in self._proc.stderr:  # type: ignore[union-attr]
                    self._stderr_buf.append(line.rstrip("\n\r"))
            except Exception:
                pass

        threading.Thread(target=_drain_stderr_loop, daemon=True).start()

        # Wait for READY — mmWave warm-up runs 500 ms simulation time,
        # which can take up to ~60 s on slow machines loading ns-3 libs.
        try:
            ready_line = self._read_line(timeout=60.0)
        except TimeoutError:
            self._kill_subprocess()
            raise RuntimeError(
                "ns3_mmwave_sim did not emit READY within 60 s. "
                "Check stderr for ns-3 error messages."
            )

        if ready_line != "READY":
            stderr_preview = self._drain_stderr()
            self._kill_subprocess()
            raise RuntimeError(
                f"ns3_mmwave_sim emitted unexpected startup line: '{ready_line}'\n"
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
            raise RuntimeError("NS3MmWaveChannel: subprocess is not running.")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                "NS3MmWaveChannel: subprocess stdin pipe broken.\n"
                f"stderr: {stderr_preview}"
            ) from exc

    def _read_line(self, timeout: float = 10.0) -> str:
        """Read one response line from the subprocess stdout (strips \\n)."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("NS3MmWaveChannel: subprocess is not running.")

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
                f"NS3MmWaveChannel: subprocess did not respond within {timeout}s.\n"
                f"stderr: {stderr_preview}"
            )
        if exc_holder[0] is not None:
            raise exc_holder[0]

        line = result[0]
        if line is None or line == "":
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                "NS3MmWaveChannel: subprocess stdout closed (process exited?).\n"
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
