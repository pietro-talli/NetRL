"""
ns3_multi_ue_channel.py
=======================
Multi-UE WiFi channel backed by a single ns-3 infrastructure BSS simulation.

Architecture
------------
A **single** ns3_wifi_multi_ue_sim subprocess simulates N UEs (STAs) all
associated with one AP.  All UEs share the same 802.11a wireless medium and
contend via CSMA/CA, producing correct multi-node uplink behaviour.

Python-side class hierarchy
---------------------------
                  ┌────────────────────────────┐
                  │  NS3WifiMultiUEBackend      │
                  │  (one instance, shared)     │
                  │  - owns the subprocess      │
                  │  - flush cache per step     │
                  │  - reset coordination       │
                  └────────────┬───────────────┘
                               │  referenced by N instances
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
  NS3WifiUEChannel(0)  NS3WifiUEChannel(1)  NS3WifiUEChannel(N-1)
  (CommChannel)        (CommChannel)        (CommChannel)
  - owns pending obs   - owns pending obs   - owns pending obs
  - ue_id = 0          - ue_id = 1          - ue_id = N-1

Each NS3WifiUEChannel instance is registered as a separate node in
CentralNode, so the existing multi-node aggregation logic is reused
without modification.

Usage
-----
    from netrl import NetworkConfig, CentralNode
    from netrl.channels.ns3_wifi_multi_ue_config import NS3WifiMultiUEConfig
    from netrl.channels.ns3_multi_ue_channel import make_multi_ue_wifi_factory
    import numpy as np

    ns3_cfg = NS3WifiMultiUEConfig(
        n_ues=3,
        distances_m=[10.0, 30.0, 60.0],
        step_duration_ms=2.0,
    )
    factory = make_multi_ue_wifi_factory(ns3_cfg)

    central = CentralNode(
        node_ids=["ue_0", "ue_1", "ue_2"],
        obs_shape=(4,),
        obs_dtype=np.float32,
        config=NetworkConfig(buffer_size=10),
        channel_factory=factory,
    )

Build the binary first:
    bash src/build_ns3_multi_ue_sim.sh

Protocol (subprocess stdin/stdout)
-----------------------------------
Python → subprocess:
    TRANSMIT <ue_id> <step_id> <pkt_size>
    FLUSH    <step_id>
    RESET
    QUIT

Subprocess → Python:
    READY
    OK
    RECV <ue_id>:<step_id> ...   (space-separated, may be empty: "RECV")
    ERROR <msg>
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from netrl.channels.comm_channel import CommChannel
from netrl.channels.network_config import NetworkConfig
from netrl.channels.ns3_wifi_multi_ue_config import NS3WifiMultiUEConfig
from netrl import netrl_multi_ue_ext

_HAS_MULTI_UE_PYBIND = True

_DEFAULT_SIM_BINARY = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "../..", "src", "ns3_wifi_multi_ue_sim"
    )
)


# ---------------------------------------------------------------------------
# Backend: owns the subprocess and is shared across all UE channel instances
# ---------------------------------------------------------------------------

class NS3WifiMultiUEBackend:
    """
    Manages the ns3_wifi_multi_ue_sim subprocess shared by all UE channels.

    Two coordination mechanisms are needed because CentralNode calls each
    CommChannel in a loop:

    Flush coordination
        flush(step) sends "FLUSH <step>" to the subprocess exactly once per
        step.  Subsequent calls for the same step return a cached result.
        CentralNode guarantees strictly increasing step values, so the cache
        is valid for the lifetime of the step.

    Reset coordination
        request_reset() is called by each NS3WifiUEChannel on reset().
        The RESET command is sent to the subprocess only after all n_ues
        channels have called request_reset(), ensuring a single clean reset.
    """

    def __init__(self, ns3_cfg: NS3WifiMultiUEConfig) -> None:
        self.ns3_cfg = ns3_cfg

        self._native = None
        self._use_pybind = _HAS_MULTI_UE_PYBIND

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_buf: deque = deque(maxlen=200)

        # Flush cache: last flushed step → {ue_id: [step_id, ...]}
        self._last_flushed_step: int = -1
        self._flush_cache: Dict[int, List[int]] = {}

        # Reset coordination
        self._reset_pending: int = 0

        if self._use_pybind:
            self._start_native_backend()
        else:
            self._start_subprocess()

    # -----------------------------------------------------------------------
    # Public API (called by NS3WifiUEChannel instances)
    # -----------------------------------------------------------------------

    def transmit(self, ue_id: int, step_id: int, pkt_size: int) -> None:
        """Send TRANSMIT <ue_id> <step_id> <pkt_size> and wait for OK."""
        if self._use_pybind:
            assert self._native is not None
            self._native.transmit(ue_id, step_id, pkt_size)
            return

        self._send_command(f"TRANSMIT {ue_id} {step_id} {pkt_size}")
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(
                f"NS3WifiMultiUEBackend transmit: unexpected response '{resp}'"
            )

    def flush(self, step: int) -> Dict[int, List[int]]:
        """
        Advance the simulation to the end of env step `step` and return
        arrived packet identifiers grouped by UE.

        The first call for a given step sends FLUSH to the subprocess and
        caches the result; subsequent calls for the same step return the
        cache.

        Returns
        -------
        Dict[ue_id -> List[step_id]]
            step_ids that arrived at the AP in this flush window, per UE.
        """
        if step == self._last_flushed_step:
            return self._flush_cache

        if step < self._last_flushed_step:
            # Should not happen under normal CentralNode usage; return empty.
            return {}

        if self._use_pybind:
            assert self._native is not None
            result: Dict[int, List[int]] = {}
            for uid, sid in self._native.flush(step):
                result.setdefault(int(uid), []).append(int(sid))
            self._last_flushed_step = step
            self._flush_cache = result
            return result

        self._send_command(f"FLUSH {step}")
        response = self._read_line()

        if not response.startswith("RECV"):
            raise RuntimeError(
                f"NS3WifiMultiUEBackend flush: unexpected response '{response}'"
            )

        # Parse "RECV ue_id:step_id ue_id:step_id ..."
        result: Dict[int, List[int]] = {}
        for part in response.split()[1:]:
            uid_str, sid_str = part.split(":")
            uid = int(uid_str)
            sid = int(sid_str)
            result.setdefault(uid, []).append(sid)

        self._last_flushed_step = step
        self._flush_cache = result
        return result

    def request_reset(self) -> None:
        """
        Called once by each NS3WifiUEChannel on reset().

        Sends RESET to the subprocess only after every UE channel has called
        this method, guaranteeing exactly one RESET per env.reset().
        """
        self._reset_pending += 1
        if self._reset_pending >= self.ns3_cfg.n_ues:
            self._reset_pending = 0
            self._last_flushed_step = -1
            self._flush_cache = {}

            if self._use_pybind:
                assert self._native is not None
                self._native.reset()
                return

            self._send_command("RESET")
            resp = self._read_line()
            if resp != "OK":
                raise RuntimeError(
                    f"NS3WifiMultiUEBackend reset: unexpected response '{resp}'"
                )

    def _start_native_backend(self) -> None:
        cfg = self.ns3_cfg
        self._native = netrl_multi_ue_ext.NS3WiFiMultiUEChannel(
            n_ues=cfg.n_ues,
            distances_m=cfg.distances_m,
            step_duration_ms=cfg.step_duration_ms,
            tx_power_dbm=cfg.tx_power_dbm,
            loss_exponent=cfg.loss_exponent,
            max_retries=cfg.max_retries,
            packet_size_bytes=cfg.packet_size_bytes,
        )

    # -----------------------------------------------------------------------
    # Subprocess management (mirrors NS3WifiChannel implementation)
    # -----------------------------------------------------------------------

    def _resolve_binary(self) -> str:
        path = self.ns3_cfg.sim_binary or _DEFAULT_SIM_BINARY
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ns3_wifi_multi_ue_sim binary not found at '{path}'.\n"
                "Build it first:\n"
                "    bash src/build_ns3_multi_ue_sim.sh\n"
                "Or set NS3WifiMultiUEConfig.sim_binary to the correct path."
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"ns3_wifi_multi_ue_sim binary at '{path}' is not executable.\n"
                "Run: chmod +x " + path
            )
        return path

    def _start_subprocess(self) -> None:
        binary = self._resolve_binary()
        cfg    = self.ns3_cfg

        distances_str = ",".join(str(d) for d in cfg.distances_m)
        cmd = [
            binary,
            f"--n-ues={cfg.n_ues}",
            f"--distances={distances_str}",
            f"--step-ms={cfg.step_duration_ms}",
            f"--tx-power={cfg.tx_power_dbm}",
            f"--loss-exp={cfg.loss_exponent}",
            f"--retries={cfg.max_retries}",
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

        # Infrastructure BSS association takes up to 500 ms warm-up; allow
        # 60 s total for the subprocess to become ready (same as mmWave).
        try:
            ready_line = self._read_line(timeout=60.0)
        except TimeoutError:
            self._kill_subprocess()
            raise RuntimeError(
                "ns3_wifi_multi_ue_sim did not emit READY within 60 s.\n"
                "Check stderr for ns-3 error messages."
            )

        if ready_line != "READY":
            preview = self._drain_stderr()
            self._kill_subprocess()
            raise RuntimeError(
                f"ns3_wifi_multi_ue_sim unexpected startup line: '{ready_line}'\n"
                f"stderr: {preview}"
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
            raise RuntimeError(
                "NS3WifiMultiUEBackend: subprocess is not running."
            )
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            preview = self._drain_stderr()
            raise RuntimeError(
                "NS3WifiMultiUEBackend: subprocess stdin pipe broken.\n"
                f"stderr: {preview}"
            ) from exc

    def _read_line(self, timeout: float = 10.0) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError(
                "NS3WifiMultiUEBackend: subprocess is not running."
            )
        result: List[Optional[str]] = [None]
        exc_holder: List[Optional[Exception]] = [None]

        def _read() -> None:
            try:
                result[0] = self._proc.stdout.readline()  # type: ignore[union-attr]
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            preview = self._drain_stderr()
            raise TimeoutError(
                f"NS3WifiMultiUEBackend: no response within {timeout} s.\n"
                f"stderr: {preview}"
            )
        if exc_holder[0] is not None:
            raise exc_holder[0]  # type: ignore[misc]

        line = result[0]
        if line is None or line == "":
            preview = self._drain_stderr()
            raise RuntimeError(
                "NS3WifiMultiUEBackend: subprocess stdout closed (process exited?).\n"
                f"Return code: {self._proc.poll()}\n"
                f"stderr: {preview}"
            )
        return line.rstrip("\n\r")

    def _drain_stderr(self, max_lines: int = 50) -> str:
        if not self._stderr_buf:
            return "<no stderr>"
        return "\n".join(list(self._stderr_buf)[-max_lines:])

    def __del__(self) -> None:
        if self._use_pybind:
            self._native = None
            return

        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send_command("QUIT")
                self._proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            self._kill_subprocess()


# ---------------------------------------------------------------------------
# Per-UE channel: CommChannel proxy delegating to the shared backend
# ---------------------------------------------------------------------------

class NS3WifiUEChannel(CommChannel):
    """
    CommChannel implementation for a single UE in the multi-UE WiFi network.

    All instances created by make_multi_ue_wifi_factory() share one
    NS3WifiMultiUEBackend (and therefore one ns-3 subprocess).  This ensures
    that all UEs contend for the same simulated wireless medium.

    Parameters
    ----------
    ue_id   : int                    Zero-based index of this UE.
    backend : NS3WifiMultiUEBackend  Shared subprocess manager.
    config  : NetworkConfig          NetRL shared config (used for metadata).
    """

    def __init__(
        self,
        ue_id: int,
        backend: NS3WifiMultiUEBackend,
        config: NetworkConfig,
    ) -> None:
        self._ue_id   = ue_id
        self._backend = backend
        self._config  = config

        # step_id → (obs_copy, sent_at_step): observations awaiting ns-3 ack
        self._pending: Dict[int, Tuple[np.ndarray, int]] = {}

    # -----------------------------------------------------------------------
    # CommChannel interface
    # -----------------------------------------------------------------------

    def transmit(
        self, obs: np.ndarray, step: int, packet_size: Optional[int] = None
    ) -> None:
        """
        Transmit `obs` from this UE at env step `step`.

        Stores the observation locally (keyed by step_id) and instructs the
        shared backend to schedule a packet send in the ns-3 simulation.
        """
        size = (
            packet_size
            if packet_size is not None
            else self._backend.ns3_cfg.packet_size_bytes
        )
        self._pending[step] = (obs.copy(), step)
        self._backend.transmit(self._ue_id, step, size)

    def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
        """
        Return observations whose packets arrived at the AP during step `step`.

        Queries the shared backend (which sends FLUSH to ns-3 at most once per
        step and caches results for subsequent calls).  Any pending
        observations older than max_pending_steps are expired.

        Returns
        -------
        List of (arrival_step, obs) tuples.
        """
        arrived_map = self._backend.flush(step)
        step_ids    = arrived_map.get(self._ue_id, [])

        result: List[Tuple[int, np.ndarray]] = []
        for sid in step_ids:
            if sid in self._pending:
                obs, _ = self._pending.pop(sid)
                result.append((step, obs))

        # Expire observations that stayed in-flight too long
        max_age = self._backend.ns3_cfg.max_pending_steps
        expired = [
            sid
            for sid, (_, sent) in self._pending.items()
            if step - sent > max_age
        ]
        for sid in expired:
            del self._pending[sid]

        return result

    def reset(self) -> None:
        """
        Clear local state and notify the backend that this UE has reset.

        The backend sends RESET to the subprocess once all n_ues channels
        have called reset(), ensuring a single coordinated simulation reset.
        """
        self._pending.clear()
        self._backend.request_reset()

    def get_channel_info(self) -> dict:
        """Return diagnostic information for this UE channel."""
        cfg = self._backend.ns3_cfg
        distances = cfg.distances_m
        dist = (
            distances[self._ue_id]
            if self._ue_id < len(distances)
            else distances[-1]
        )
        return {
            "state":            "NS3_WIFI_MULTI_UE",
            "ue_id":            self._ue_id,
            "n_ues":            cfg.n_ues,
            "pending_count":    len(self._pending),
            "distance_m":       dist,
            "step_duration_ms": cfg.step_duration_ms,
            "tx_power_dbm":     cfg.tx_power_dbm,
            "loss_exponent":    cfg.loss_exponent,
            "max_retries":      cfg.max_retries,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_multi_ue_wifi_factory(
    ns3_cfg: NS3WifiMultiUEConfig,
) -> Callable[[NetworkConfig], CommChannel]:
    """
    Create a channel_factory callable for use with CentralNode.

    A **single** ns3_wifi_multi_ue_sim subprocess is started immediately.
    Each call to the returned factory creates one NS3WifiUEChannel proxy
    (indexed sequentially: 0, 1, 2, …) that delegates to the shared backend.

    The factory must be called exactly ns3_cfg.n_ues times — once per
    node_id registered with CentralNode — so that len(node_ids) == n_ues.

    Parameters
    ----------
    ns3_cfg : NS3WifiMultiUEConfig
        Physical-layer and network configuration.

    Returns
    -------
    Callable[[NetworkConfig], CommChannel]
        A factory suitable for CentralNode's channel_factory parameter.

    Example
    -------
    ::

        ns3_cfg = NS3WifiMultiUEConfig(
            n_ues=3,
            distances_m=[10.0, 30.0, 60.0],
            step_duration_ms=2.0,
        )
        factory = make_multi_ue_wifi_factory(ns3_cfg)

        central = CentralNode(
            node_ids=["ue_0", "ue_1", "ue_2"],
            obs_shape=(4,),
            obs_dtype=np.float32,
            config=NetworkConfig(buffer_size=10),
            channel_factory=factory,
        )
    """
    ns3_cfg.validate()
    backend: NS3WifiMultiUEBackend = NS3WifiMultiUEBackend(ns3_cfg)
    counter: List[int] = [0]

    def _factory(net_cfg: NetworkConfig) -> CommChannel:
        ue_id = counter[0]
        counter[0] += 1
        if ue_id >= ns3_cfg.n_ues:
            raise ValueError(
                f"make_multi_ue_wifi_factory: factory called for UE index "
                f"{ue_id} but NS3WifiMultiUEConfig.n_ues={ns3_cfg.n_ues}. "
                "Ensure len(node_ids) passed to CentralNode equals n_ues."
            )
        return NS3WifiUEChannel(ue_id, backend, net_cfg)

    return _factory
