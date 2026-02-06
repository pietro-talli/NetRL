"""Example Python-side RL loop runner."""

from netrl.loop import RLEnvironmentLoop, TcpJsonBridge


class EchoAgent:
    def act(self, observation: dict) -> dict:
        return {"echo": observation, "action": "noop"}


def main() -> None:
    bridge = TcpJsonBridge(host="127.0.0.1", port=5555)
    agent = EchoAgent()
    loop = RLEnvironmentLoop(bridge=bridge, agent=agent)
    try:
        while True:
            loop.step()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
