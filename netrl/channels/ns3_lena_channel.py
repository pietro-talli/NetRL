"""
ns3_lena_channel.py
===================
NS3LenaChannel — a CommChannel backed by a 5G-LENA ns-3 simulation
running as a persistent subprocess.

Protocol (stdin/stdout, line-oriented)
--------------------------------------
    TRANSMIT <step_id> <size>
    FLUSH    <step_id>
    RESET
    QUIT

Responses:
    READY
    OK
    RECV <id1> <id2> ...
    ERROR <msg>
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
from netrl.channels.ns3_lena_config import NS3LenaConfig

_DEFAULT_SIM_BINARY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../..", "src", "ns3_lena_sim")
)


class NS3LenaChannel(CommChannel):
    """CommChannel implementation backed by an ns-3 5G-LENA NR simulation."""

    def __init__(
        self,
        config: NetworkConfig,
        ns3_config: Optional[NS3LenaConfig] = None,
    ) -> None:
        self._config = config
        self._ns3_cfg: NS3LenaConfig = ns3_config or NS3LenaConfig()
        self._ns3_cfg.validate()

        self._pending: Dict[int, Tuple[np.ndarray, int]] = {}
        self._arrived: deque = deque()

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_buf: deque = deque(maxlen=200)
        self._start_subprocess()

    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: Optional[int] = None) -> None:
        size = packet_size if packet_size is not None else self._ns3_cfg.packet_size_bytes
        self._pending[step] = (obs.copy(), step)
        self._send_command(f"TRANSMIT {step} {size}")
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(
                f"NS3LenaChannel transmit: unexpected response '{resp}'"
            )

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        self._send_command(f"FLUSH {step}")
        response = self._read_line()

        if not response.startswith("RECV"):
            raise RuntimeError(
                f"NS3LenaChannel flush: unexpected response '{response}'"
            )

        parts = response.split()
        for part in parts[1:]:
            sid = int(part)
            if sid in self._pending:
                obs, _ = self._pending.pop(sid)
                self._arrived.append((step, obs))

        expired = [
            sid
            for sid, (_, sent_step) in self._pending.items()
            if step - sent_step > self._ns3_cfg.max_pending_steps
        ]
        for sid in expired:
            self._pending.pop(sid)

        result: List[Tuple[int, np.ndarray]] = []
        while self._arrived and self._arrived[0][0] <= step:
            result.append(self._arrived.popleft())

        return result

    def reset(self) -> None:
        self._pending.clear()
        self._arrived.clear()
        self._send_command("RESET")
        resp = self._read_line(timeout=60.0)
        if resp != "OK":
            raise RuntimeError(
                f"NS3LenaChannel reset: unexpected response '{resp}'"
            )

    def get_channel_info(self) -> dict:
        return {
            "state": "NS3_LENA",
            "pending_count": len(self._pending),
            "arrived_buffered": len(self._arrived),
            "distance_m": self._ns3_cfg.distance_m,
            "frequency_ghz": self._ns3_cfg.frequency_ghz,
            "bandwidth_ghz": self._ns3_cfg.bandwidth_ghz,
            "ue_tx_power_dbm": self._ns3_cfg.ue_tx_power_dbm,
            "gnb_tx_power_dbm": self._ns3_cfg.gnb_tx_power_dbm,
            "scenario": self._ns3_cfg.scenario,
            "numerology": self._ns3_cfg.numerology,
            "step_duration_ms": self._ns3_cfg.step_duration_ms,
        }

    def _resolve_binary(self) -> str:
        path = self._ns3_cfg.sim_binary or _DEFAULT_SIM_BINARY
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ns3_lena_sim binary not found at '{path}'.\n"
                "Build it first:\n"
                "    bash src/build_ns3_lena_sim.sh\n"
                "Or set NS3LenaConfig.sim_binary to the correct path."
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"ns3_lena_sim binary at '{path}' is not executable.\n"
                "Run: chmod +x " + path
            )
        return path

    def _start_subprocess(self) -> None:
        binary = self._resolve_binary()
        cfg = self._ns3_cfg

        cmd = [
            binary,
            f"--step-ms={cfg.step_duration_ms}",
            f"--distance={cfg.distance_m}",
            f"--freq={cfg.frequency_ghz * 1e9}",
            f"--bandwidth={cfg.bandwidth_ghz * 1e9}",
            f"--ue-tx-power={cfg.ue_tx_power_dbm}",
            f"--gnb-tx-power={cfg.gnb_tx_power_dbm}",
            f"--scenario={cfg.scenario}",
            f"--numerology={cfg.numerology}",
            f"--shadowing={int(cfg.shadowing_enabled)}",
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

        def _drain_stderr_loop() -> None:
            try:
                for line in self._proc.stderr:  # type: ignore[union-attr]
                    self._stderr_buf.append(line.rstrip("\n\r"))
            except Exception:
                pass

        threading.Thread(target=_drain_stderr_loop, daemon=True).start()

        try:
            ready_line = self._read_line(timeout=60.0)
        except TimeoutError:
            self._kill_subprocess()
            raise RuntimeError(
                "ns3_lena_sim did not emit READY within 60 s. "
                "Check stderr for ns-3 error messages."
            )

        if ready_line != "READY":
            stderr_preview = self._drain_stderr()
            self._kill_subprocess()
            raise RuntimeError(
                f"ns3_lena_sim emitted unexpected startup line: '{ready_line}'\n"
                f"stderr: {stderr_preview}"
            )

    def _kill_subprocess(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _send_command(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("NS3LenaChannel: subprocess is not running.")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                "NS3LenaChannel: subprocess stdin pipe broken.\n"
                f"stderr: {stderr_preview}"
            ) from exc

    def _read_line(self, timeout: float = 10.0) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("NS3LenaChannel: subprocess is not running.")

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
                f"NS3LenaChannel: subprocess did not respond within {timeout}s.\n"
                f"stderr: {stderr_preview}"
            )
        if exc_holder[0] is not None:
            raise exc_holder[0]

        line = result[0]
        if line is None or line == "":
            stderr_preview = self._drain_stderr()
            raise RuntimeError(
                "NS3LenaChannel: subprocess stdout closed (process exited?).\n"
                f"Return code: {self._proc.poll()}\n"
                f"stderr: {stderr_preview}"
            )
        return line.rstrip("\n\r")

    def _drain_stderr(self, max_lines: int = 50) -> str:
        if not self._stderr_buf:
            return "<no stderr>"
        lines = list(self._stderr_buf)[-max_lines:]
        return "\n".join(lines)

    def __del__(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send_command("QUIT")
                self._proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            self._kill_subprocess()
