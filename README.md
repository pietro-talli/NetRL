# NetRL — Networked Reinforcement Learning Simulation Platform

NetRL wraps any [Gymnasium](https://gymnasium.farama.org/) environment and simulates a **noisy communication channel** between the agent and the environment. At every `step()`, the raw observation is transmitted through a configurable channel backend (loss, delay, retransmissions). The agent receives a **sliding-window buffer** of past observations together with a boolean mask indicating which slots actually arrived.

Two channel backends are available:

| Backend | Model | Config class |
|---|---|---|
| **Gilbert-Elliott** (default) | Two-state Markov chain with configurable loss per state and fixed delay | `NetworkConfig` |
| **ns-3 802.11a WiFi** | Full MAC/PHY simulation via ns-3 — CSMA/CA, retransmissions, path-loss | `NS3WifiConfig` |

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Python package and GE channel (pybind11)](#1-python-package-and-ge-channel-pybind11)
   - [ns-3 WiFi binary](#2-ns-3-wifi-binary)
3. [Quick Start](#quick-start)
   - [Gilbert-Elliott channel](#gilbert-elliott-channel)
   - [ns-3 WiFi channel](#ns-3-wifi-channel)
4. [Observation Space](#observation-space)
5. [Configuration Reference](#configuration-reference)
   - [NetworkConfig](#networkconfig)
   - [NS3WifiConfig](#ns3wificonfig)
6. [NetworkedEnv API](#networkedenv-api)
7. [Per-step Packet Size](#per-step-packet-size)
8. [Building the ns-3 Binary](#building-the-ns-3-binary)
   - [Prerequisites](#prerequisites)
   - [Build script](#build-script)
   - [Build options](#build-options)
   - [Manual compilation](#manual-compilation)
   - [Verification](#verification)
9. [Advanced: CentralNode and Custom Channels](#advanced-centralnode-and-custom-channels)
10. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python ≥ 3.10
- GCC ≥ 10 or Clang ≥ 11 (C++17 for the pybind11 extension; C++20 for the ns-3 binary when using pip-installed ns-3 ≥ 3.43)
- One of:
  - `pip install ns3` (ns-3 ≥ 3.43, headers + shared libs, C++20)
  - A compiled ns-3-mmwave source build (ns-3 3.42, C++17)

---

## Installation

### 1. Python package and GE channel (pybind11)

The Gilbert-Elliott channel is implemented as a C++ pybind11 extension (`netcomm`). Install it together with the Python package:

```bash
pip install pybind11 numpy gymnasium
pip install -e .
```

This builds `netcomm.cpython-*.so` in place and installs the `netrl` package in editable mode. To build the extension in-place without installing:

```bash
python setup.py build_ext --inplace
```

> The GE channel (default backend) requires this step. The `PerfectChannel` baseline does **not**.

### 2. ns-3 WiFi binary

The ns-3 backend runs as a **persistent subprocess**. Compile it once before use:

```bash
bash src/build_ns3_sim.sh
```

The script auto-detects your ns-3 installation, compiles `src/ns3_wifi_sim.cc`, and writes the binary to `src/ns3_wifi_sim`. See [Building the ns-3 Binary](#building-the-ns-3-binary) for full details.

---

## Quick Start

### Gilbert-Elliott channel

```python
import gymnasium as gym
from netrl import NetworkedEnv, NetworkConfig

config = NetworkConfig(
    p_gb=0.1,        # Good -> Bad transition probability per step
    p_bg=0.3,        # Bad  -> Good transition probability per step
    loss_good=0.01,  # packet loss probability in Good state
    loss_bad=0.20,   # packet loss probability in Bad state
    delay_steps=3,   # fixed one-way delay in env steps
    buffer_size=10,  # observation window length
    seed=42,
)

env = NetworkedEnv(gym.make("CartPole-v1"), config)
obs, info = env.reset()

# obs["observations"].shape == (10, 4)   — buffer of last 10 obs
# obs["recv_mask"].shape    == (10,)     — True where a packet arrived

for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])   # "GOOD" or "BAD"
    print(info["arrived_this_step"])       # True / False
    if term or trunc:
        obs, info = env.reset()
```

### ns-3 WiFi channel

```python
import gymnasium as gym
from netrl import NetworkedEnv, NetworkConfig, NS3WifiConfig

env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10),
    channel_config=NS3WifiConfig(
        distance_m=20.0,         # STA-to-AP distance
        step_duration_ms=5.0,    # 5 ms of ns-3 time per env step
        tx_power_dbm=20.0,
        loss_exponent=3.0,
        max_retries=7,
        packet_size_bytes=256,
    ),
)

obs, info = env.reset()

for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])   # "NS3_WIFI"
    if term or trunc:
        obs, info = env.reset()
```

> The ns-3 simulation is **persistent across steps** — MAC-layer state (backoff counters, retry timers) carries over between steps, giving temporally correlated, realistic channel behaviour. The simulation is only rebuilt on `env.reset()`.

---

## Observation Space

`NetworkedEnv` replaces the wrapped env's `Box` observation space with:

```python
gymnasium.spaces.Dict({
    "observations": Box(
        shape=(buffer_size, *original_obs_shape),
        dtype=original_obs_dtype,
    ),
    "recv_mask": MultiBinary(buffer_size),
})
```

- `observations[-1]` is the **most recent** slot; `observations[0]` is the oldest.
- Slots where no packet arrived (dropped or not yet delivered) are zero-filled with `recv_mask = False`.
- The buffer always has exactly `buffer_size` entries, providing a **fixed-shape input** for neural network policies.

The `info` dict returned by `step()` is augmented with:

| Key | Type | Description |
|---|---|---|
| `"channel_info"` | `dict` | Diagnostic snapshot from the channel backend |
| `"arrived_this_step"` | `bool` | `True` if at least one packet arrived during this step |

**GE `channel_info` keys:** `state` (`"GOOD"` / `"BAD"`), `pending_count`

**ns-3 `channel_info` keys:** `state` (`"NS3_WIFI"`), `pending_count`, `arrived_buffered`, `distance_m`, `step_duration_ms`, `tx_power_dbm`, `loss_exponent`, `max_retries`

---

## Configuration Reference

### NetworkConfig

Controls the Gilbert-Elliott channel model and the observation buffer size.

```python
from netrl import NetworkConfig

config = NetworkConfig(
    p_gb        = 0.1,   # Prob. of Good -> Bad transition per step  [0, 1]
    p_bg        = 0.3,   # Prob. of Bad  -> Good transition per step [0, 1]
    loss_good   = 0.01,  # Packet loss probability in Good state     [0, 1]
    loss_bad    = 0.20,  # Packet loss probability in Bad state      [0, 1]
    delay_steps = 3,     # Fixed one-way delay in env steps          [>= 0]
    buffer_size = 10,    # Observation window length                 [>= 1]
    seed        = 42,    # RNG seed for the C++ backend
)
```

| Field | Default | Description |
|---|---|---|
| `p_gb` | `0.1` | Good → Bad transition probability per step |
| `p_bg` | `0.3` | Bad → Good transition probability per step |
| `loss_good` | `0.01` | Packet loss probability in the Good state |
| `loss_bad` | `0.20` | Packet loss probability in the Bad state |
| `delay_steps` | `3` | Fixed one-way propagation delay in env steps |
| `buffer_size` | `10` | Sliding-window length; sets the first dim of `observations` |
| `seed` | `42` | RNG seed forwarded to the C++ backend |

Steady-state probability of being in the Bad state: `p_gb / (p_gb + p_bg)`

> When using the ns-3 backend, only `buffer_size` and `seed` from `NetworkConfig` are used. The GE-specific fields (`p_gb`, `p_bg`, `loss_good`, `loss_bad`, `delay_steps`) are ignored — WiFi physics govern loss and delay.

### NS3WifiConfig

Controls the ns-3 802.11a physical layer and the subprocess.

```python
from netrl import NS3WifiConfig

ns3_config = NS3WifiConfig(
    distance_m        = 10.0,  # STA-to-AP Euclidean distance in metres
    step_duration_ms  = 1.0,   # Width of one env step in ns-3 sim time (ms)
    tx_power_dbm      = 20.0,  # STA transmit power in dBm
    loss_exponent     = 3.0,   # Log-distance path-loss exponent
    max_retries       = 7,     # Maximum MAC-layer retransmission attempts
    packet_size_bytes = 64,    # Default UDP payload in bytes (min 4)
    max_pending_steps = 200,   # Steps before unacked packet is declared lost
    sim_binary        = "",    # Path to binary; "" = auto-detect
)
```

| Field | Default | Description |
|---|---|---|
| `distance_m` | `10.0` | STA-to-AP distance in metres. Larger → higher path-loss → more retransmissions and drops. |
| `step_duration_ms` | `1.0` | ns-3 simulation time per env step (ms). See table below. |
| `tx_power_dbm` | `20.0` | Transmit power in dBm. |
| `loss_exponent` | `3.0` | Log-distance path-loss exponent: 2 = free-space, 3 = outdoor, 4 = dense indoor. |
| `max_retries` | `7` | MAC retry limit before a frame is dropped. Each retry adds ~3–5 ms of delay. |
| `packet_size_bytes` | `64` | Default probe packet payload in bytes. Overridable per step. |
| `max_pending_steps` | `200` | Python-side expiry: packets older than this many steps are discarded as lost. |
| `sim_binary` | `""` | Absolute path to `ns3_wifi_sim`. Auto-detected as `<project_root>/src/ns3_wifi_sim` when empty. |

**Choosing `step_duration_ms`:**

| Value | Effect |
|---|---|
| 1 ms | Fine-grained; one step ≈ one MAC frame. Retransmissions span multiple steps, producing realistic multi-step delivery delays. |
| 5–10 ms | Several retransmissions fit within a step. Most packets arrive this step or are dropped. Low delay variance. |
| 20+ ms | Very coarse. Nearly all packets arrive in the same step or drop. Minimal delay variation. |

---

## NetworkedEnv API

```python
class NetworkedEnv(gymnasium.Wrapper):

    def __init__(
        self,
        env: gymnasium.Env,
        config: NetworkConfig,
        channel_config: Optional[NS3WifiConfig] = None,
        node_id: str = "agent_0",
    )
```

| Parameter | Description |
|---|---|
| `env` | Base gymnasium environment. Must have a `Box` observation space. |
| `config` | `NetworkConfig`. Controls buffer size and GE channel parameters. |
| `channel_config` | `None` → GE channel. `NS3WifiConfig(...)` → ns-3 WiFi. |
| `node_id` | Node identifier string. Only change for non-default multi-agent setups. |

```python
# Reset — returns zero-filled buffer (no packet transmitted yet)
obs, info = env.reset(seed=None, options=None)

# Step
obs, reward, terminated, truncated, info = env.step(action, packet_size=None)

# Properties
env.step_count    # int  — current 0-indexed step counter
env.config        # NetworkConfig
env.central_node  # CentralNode — direct access for advanced use
```

---

## Per-step Packet Size

The ns-3 backend models full 802.11a MAC, so packet size directly affects **transmission time** and **drop probability**. You can override it per step:

```python
# Small control packet
obs, *_ = env.step(action, packet_size=64)

# Large observation payload — longer TX time, higher collision chance
obs, *_ = env.step(action, packet_size=2048)

# Use the default from NS3WifiConfig.packet_size_bytes
obs, *_ = env.step(action)
```

> `packet_size` is silently ignored by the GE and Perfect backends.

---

## Building the ns-3 Binary

### Prerequisites

You need one of the following ns-3 installations.

**Option A — pip (recommended, ns-3 ≥ 3.43):**

```bash
pip install ns3
```

Installs ns-3 headers and shared libraries into your Python environment. The build script compiles with `-std=c++20`.

**Option B — ns-3-mmwave source build (ns-3 3.42):**

The build script falls back to a compiled ns-3-mmwave tree at `/path/to/ns3/build`. Uses `-std=c++17`.

In both cases, `g++` must be available and support C++17 / C++20 respectively.

### Build script

```bash
bash src/build_ns3_sim.sh
```

The script:
1. Detects the ns-3 installation (pip first, ns-3-mmwave fallback)
2. Compiles `src/ns3_wifi_sim.cc` with all required ns-3 include and library paths
3. Embeds the library search path with `-Wl,-rpath` so the binary is self-contained
4. Runs a smoke test to verify the binary responds correctly
5. Places the output at `src/ns3_wifi_sim`

Expected output:

```
=== NetRL ns3 WiFi simulation build ===
  Source : .../src/ns3_wifi_sim.cc
  ns3    : pip-installed ns3-44 at ...

Compiling with -std=c++20 -O2 ...
Built: .../src/ns3_wifi_sim

Smoke test (QUIT) ...
Smoke test PASSED
```

### Build options

```bash
bash src/build_ns3_sim.sh            # Release build (-O2, default)
bash src/build_ns3_sim.sh --release  # Explicit release
bash src/build_ns3_sim.sh --debug    # Debug build (-O0 -g)
```

### Manual compilation

If the build script does not work for your setup, compile manually. The binary requires these ns-3 modules: `core`, `network`, `internet`, `wifi`, `mobility`, `propagation`.

**With pip-installed ns-3:**

```bash
NS3_PREFIX=$(python -c "import ns3, os; print(os.path.dirname(ns3.__file__))")

g++ -std=c++20 -O2 \
    -I${NS3_PREFIX}/include \
    -L${NS3_PREFIX}/lib64 \
    -Wl,-rpath,${NS3_PREFIX}/lib64 \
    src/ns3_wifi_sim.cc \
    -lns3-dev-core-default \
    -lns3-dev-network-default \
    -lns3-dev-internet-default \
    -lns3-dev-wifi-default \
    -lns3-dev-mobility-default \
    -lns3-dev-propagation-default \
    -o src/ns3_wifi_sim
```

> Library names vary by version (e.g. `libns3-44-core-default.so`). Find yours with:
> ```bash
> ls ${NS3_PREFIX}/lib64/libns3*core*
> ```

**With ns-3-mmwave source build:**

```bash
NS3_BUILD=path/to/ns3/build

g++ -std=c++17 -O2 \
    -I${NS3_BUILD}/include \
    -L${NS3_BUILD}/lib \
    -Wl,-rpath,${NS3_BUILD}/lib \
    src/ns3_wifi_sim.cc \
    -lns3.42-core-default \
    -lns3.42-network-default \
    -lns3.42-internet-default \
    -lns3.42-wifi-default \
    -lns3.42-mobility-default \
    -lns3.42-propagation-default \
    -o src/ns3_wifi_sim
```

### Verification

```bash
printf 'QUIT\n' | src/ns3_wifi_sim
# Expected single line of output: READY
```

---

## Advanced: CentralNode and Custom Channels

`CentralNode` is the underlying aggregator; `NetworkedEnv` creates one internally. You can use it directly for **multi-agent** scenarios or custom wiring:

```python
from netrl import CentralNode, NetworkConfig, GEChannel
import numpy as np

config = NetworkConfig(buffer_size=10)

central = CentralNode(
    node_ids=["agent_0", "agent_1"],
    obs_shape=(4,),
    obs_dtype=np.float32,
    config=config,
    channel_factory=GEChannel,   # or PerfectChannel, or a lambda for NS3
)

# Per step (step counter managed by the caller):
central.receive_from("agent_0", obs_0, step)
central.receive_from("agent_1", obs_1, step, packet_size=512)
arrived = central.flush_and_update(step)  # {node_id -> obs | None}

buf_0, mask_0 = central.get_buffer("agent_0")  # shape (10, 4), (10,)
buf_1, mask_1 = central.get_buffer("agent_1")

# On reset:
central.reset()
```

**Implementing a custom channel backend:**

Subclass `CommChannel` and implement four methods:

```python
from netrl import CommChannel
import numpy as np

class MyChannel(CommChannel):

    def __init__(self, config):
        ...

    def transmit(self, obs: np.ndarray, step: int,
                 packet_size: int | None = None) -> None:
        # Schedule obs for delivery. Apply your loss / delay model.
        ...

    def flush(self, step: int) -> list[tuple[int, np.ndarray]]:
        # Return [(arrival_step, obs), ...] for all packets due <= step.
        ...

    def reset(self) -> None:
        # Clear pending state. Do NOT re-seed the RNG.
        ...

    def get_channel_info(self) -> dict:
        # Must include at minimum: {"state": str, "pending_count": int}
        ...
```

Pass it through `CentralNode`:

```python
central = CentralNode(
    node_ids=["agent_0"],
    obs_shape=(4,),
    obs_dtype=np.float32,
    config=config,
    channel_factory=lambda cfg: MyChannel(cfg),
)
```

---

## Troubleshooting

**`ImportError: netcomm C++ extension not found`**

The pybind11 extension has not been built. Run:
```bash
pip install -e .
# or
python setup.py build_ext --inplace
```

**`RuntimeError: ns3_wifi_sim binary not found`**

The ns-3 binary has not been compiled. Run:
```bash
bash src/build_ns3_sim.sh
```

**Build script fails with a `glib-2.0` or CMake path error**

This is a known issue with some pip ns-3 installations where the CMake config references non-existent system paths. The build script bypasses CMake and calls `g++` directly to avoid this. Ensure you are using the current `src/build_ns3_sim.sh`.
