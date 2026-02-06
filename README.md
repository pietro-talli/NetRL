# NetRL
Networked Reinforcement Learning - A simulation platform for optimization of Multi-View and Multi-Agent Reinforcement Learning

## Goal
NetRL keeps the reinforcement-learning loop in Python while delegating network
behavior to an ns-3/C++ simulation. A lightweight bridge carries observations
from an ns-3 observer to a Python agent, and returns actions back to ns-3.

## Architecture
- **Python RL loop**: runs the agent policy and training logic.
- **Observer (ns-3)**: captures network state (queue length, RTT, throughput).
- **Bridge**: transports serialized observations/actions between ns-3 and Python.

```
ns-3 Observer -> Bridge -> Python Agent -> Bridge -> ns-3 Action Applier
```

## Repository layout
- `netrl/`: Python interfaces and the core RL loop.
- `scripts/`: example runner for the Python RL loop.
- `ns3/`: C++ skeleton for the ns-3 side bridge.
- `proto/`: optional protobuf definitions for RPC-based bridges.

## Getting started (Python loop)
1. Start a TCP bridge server in your ns-3 simulation (see `ns3/netrl-bridge.cc`).
2. Run the Python loop:

```bash
python scripts/run_python_loop.py
```

## Bridge options
- **TCP + JSON** (default in `TcpJsonBridge`): simple to prototype.
- **gRPC + protobuf** (`proto/netrl.proto`): stronger schema guarantees.
