"""Core RL loop abstractions for NetRL.

The RL loop stays in Python while the network simulation (ns-3/C++)
communicates via a bridge. The bridge exchanges observations and actions
between an Observer inside ns-3 and a Python Agent.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Protocol


class Agent(Protocol):
    """Defines the agent API for the Python RL loop."""

    def act(self, observation: dict) -> dict:
        """Return an action dictionary for a given observation."""


class Observer(Protocol):
    """Defines the observer API for ns-3 to provide state."""

    def observe(self) -> dict:
        """Return the latest observation dictionary."""


class NetworkBridge(Protocol):
    """Bridge between Python RL loop and ns-3 simulation."""

    def recv_observation(self) -> dict:
        """Receive a new observation from ns-3."""

    def send_action(self, action: dict) -> None:
        """Send an action to ns-3."""


@dataclass
class RLEnvironmentLoop:
    """Runs the RL loop, keeping learning logic in Python."""

    bridge: NetworkBridge
    agent: Agent

    def step(self) -> None:
        observation = self.bridge.recv_observation()
        action = self.agent.act(observation)
        self.bridge.send_action(action)


class TcpJsonBridge:
    """Minimal JSON-over-TCP bridge for ns-3 interop."""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0) -> None:
        self._socket = socket.create_connection((host, port), timeout=timeout_s)
        self._buffer = bytearray()

    def recv_observation(self) -> dict:
        while b"\n" not in self._buffer:
            data = self._socket.recv(4096)
            if not data:
                raise ConnectionError("ns-3 bridge closed")
            self._buffer.extend(data)
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        return json.loads(line.decode("utf-8"))

    def send_action(self, action: dict) -> None:
        payload = json.dumps(action).encode("utf-8") + b"\n"
        self._socket.sendall(payload)

    def close(self) -> None:
        self._socket.close()
