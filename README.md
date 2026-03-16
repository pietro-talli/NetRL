# NetRL — Networked Reinforcement Learning Simulation Platform

NetRL wraps any [Gymnasium](https://gymnasium.farama.org/) environment and simulates a **noisy communication channel** between the agent and the environment. At every `step()`, the raw observation is transmitted through a configurable channel backend (loss, delay, retransmissions). The agent receives a **sliding-window buffer** of past observations together with a boolean mask indicating which slots actually arrived.

Three channel backends are available:

| Backend | Model | Config class |
|---|---|---|
| **Gilbert-Elliott** (default) | Two-state Markov chain with configurable loss per state and fixed delay | `NetworkConfig` |
| **ns-3 802.11a WiFi** | Full MAC/PHY simulation via ns-3 — CSMA/CA, retransmissions, path-loss | `NS3WifiConfig` |
| **ns-3 5G mmWave** | Full EPC/NR simulation via ns-3-mmwave — 3GPP TR 38.901 path-loss, HARQ, RLC | `NS3MmWaveConfig` |

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Python package and GE channel (pybind11)](#1-python-package-and-ge-channel-pybind11)
   - [ns-3 WiFi binary](#2-ns-3-wifi-binary)
   - [ns-3 5G mmWave binary](#3-ns-3-5g-mmwave-binary)
3. [Quick Start](#quick-start)
   - [Gilbert-Elliott channel](#gilbert-elliott-channel)
   - [ns-3 WiFi channel](#ns-3-wifi-channel)
   - [ns-3 5G mmWave channel](#ns-3-5g-mmwave-channel)
4. [Observation Space](#observation-space)
5. [Configuration Reference](#configuration-reference)
   - [NetworkConfig](#networkconfig)
   - [NS3WifiConfig](#ns3wificonfig)
   - [NS3MmWaveConfig](#ns3mmwaveconfig)
6. [NetworkedEnv API](#networkedenv-api)
7. [Per-step Packet Size](#per-step-packet-size)
8. [Building the ns-3 WiFi Binary](#building-the-ns-3-wifi-binary)
   - [Prerequisites](#prerequisites)
   - [Build script](#build-script)
   - [Build options](#build-options)
   - [Manual compilation](#manual-compilation)
   - [Verification](#verification)
9. [Building the ns-3 5G mmWave Binary](#building-the-ns-3-5g-mmwave-binary)
   - [Prerequisites (mmWave)](#prerequisites-mmwave)
   - [Build script (mmWave)](#build-script-mmwave)
   - [Build options (mmWave)](#build-options-mmwave)
   - [Manual compilation (mmWave)](#manual-compilation-mmwave)
   - [Verification (mmWave)](#verification-mmwave)
10. [Advanced: CentralNode and Custom Channels](#advanced-centralnode-and-custom-channels)
11. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python ≥ 3.10
- GCC ≥ 10 or Clang ≥ 11
  - C++17 for the pybind11 extension and the ns-3 WiFi binary (pip ns-3 ≥ 3.43 only)
  - **C++20** for the ns-3 mmWave binary (ns-3-mmwave 3.42 uses `std::remove_cvref_t`)
- One of:
  - `pip install ns3` (ns-3 ≥ 3.43, headers + shared libs) — for the WiFi backend
  - A compiled **ns-3-mmwave** source build at `/path/to/ns3-mmwave/build` (ns-3 3.42) — required for the 5G mmWave backend

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

The script auto-detects your ns-3 installation, compiles `src/ns3_wifi_sim.cc`, and writes the binary to `src/ns3_wifi_sim`. See [Building the ns-3 WiFi Binary](#building-the-ns-3-wifi-binary) for full details.

### 3. ns-3 5G mmWave binary

The mmWave backend requires a separate binary built against [ns-3-mmwave](https://github.com/nyuwireless-unipd/ns3-mmwave) (ns-3 3.42). Compile it once before use:

```bash
bash src/build_ns3_mmwave_sim.sh
```

The script expects ns-3-mmwave to be built at `/home/dianalab/Projects/ns3-mmwave/build` (edit the `NS3_MMWAVE_BUILD` variable at the top of the script to change the path). See [Building the ns-3 5G mmWave Binary](#building-the-ns-3-5g-mmwave-binary) for full details.

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

### ns-3 5G mmWave channel

```python
import gymnasium as gym
from netrl import NetworkedEnv, NetworkConfig, NS3MmWaveConfig

env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10),
    channel_config=NS3MmWaveConfig(
        distance_m=50.0,           # UE-to-eNB distance
        frequency_ghz=28.0,        # 28 GHz (n257/n261 band)
        bandwidth_ghz=0.2,         # 200 MHz component carrier
        tx_power_dbm=23.0,         # UE transmit power
        scenario="UMa",            # Urban Macro (3GPP TR 38.901)
        harq_enabled=True,
        step_duration_ms=1.0,
        packet_size_bytes=64,
    ),
)

obs, info = env.reset()

for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])   # "NS3_MMWAVE"
    if term or trunc:
        obs, info = env.reset()
```

> The mmWave simulation models a full 5G EPC stack (UE → eNB → SGW/PGW → remote host) with 3GPP TR 38.901 path-loss, HARQ retransmissions, and configurable RLC mode. The first `reset()` call waits up to 60 s for the EPC to initialise — subsequent resets are faster.

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

**ns-3 WiFi `channel_info` keys:** `state` (`"NS3_WIFI"`), `pending_count`, `arrived_buffered`, `distance_m`, `step_duration_ms`, `tx_power_dbm`, `loss_exponent`, `max_retries`

**ns-3 mmWave `channel_info` keys:** `state` (`"NS3_MMWAVE"`), `pending_count`, `arrived_buffered`, `distance_m`, `frequency_ghz`, `bandwidth_ghz`, `tx_power_dbm`, `enb_tx_power_dbm`, `noise_figure_db`, `scenario`, `harq_enabled`, `rlc_am_enabled`, `step_duration_ms`

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

> When using the ns-3 WiFi or 5G mmWave backend, only `buffer_size` and `seed` from `NetworkConfig` are used. The GE-specific fields (`p_gb`, `p_bg`, `loss_good`, `loss_bad`, `delay_steps`) are ignored — the ns-3 physical simulation governs loss and delay.

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

### NS3MmWaveConfig

Controls the ns-3 5G mmWave EPC simulation. Requires the `ns3_mmwave_sim` binary (see [Building the ns-3 5G mmWave Binary](#building-the-ns-3-5g-mmwave-binary)).

```python
from netrl import NS3MmWaveConfig

ns3_mmwave_config = NS3MmWaveConfig(
    distance_m          = 50.0,    # UE-to-eNB distance in metres
    frequency_ghz       = 28.0,    # Carrier frequency (GHz). Common: 28, 39
    bandwidth_ghz       = 0.2,     # Component carrier bandwidth (GHz)
    tx_power_dbm        = 23.0,    # UE transmit power (dBm)
    enb_tx_power_dbm    = 30.0,    # eNB transmit power (dBm)
    noise_figure_db     = 9.0,     # UE receiver noise figure (dB)
    enb_noise_figure_db = 5.0,     # eNB receiver noise figure (dB)
    scenario            = "UMa",   # 3GPP TR 38.901 propagation scenario
    harq_enabled        = True,    # Hybrid ARQ with incremental redundancy
    rlc_am_enabled      = False,   # RLC Acknowledged Mode (adds extra delay)
    packet_size_bytes   = 64,      # Default UDP payload in bytes (min 4)
    step_duration_ms    = 1.0,     # Width of one env step in ns-3 sim time (ms)
    max_pending_steps   = 500,     # Steps before unacked packet is declared lost
    sim_binary          = "",      # Path to binary; "" = auto-detect
)
```

| Field | Default | Description |
|---|---|---|
| `distance_m` | `50.0` | UE-to-eNB Euclidean distance in metres. At 28 GHz (UMa), link failure begins ~200 m. |
| `frequency_ghz` | `28.0` | Carrier frequency in GHz. Common 5G mmWave bands: 26.5–29.5 (n257/n261), 37–40 (n260). |
| `bandwidth_ghz` | `0.2` | Component carrier bandwidth in GHz (200 MHz). Typical NR: 0.05–0.4 GHz. |
| `tx_power_dbm` | `23.0` | UE transmit power in dBm. Typical 5G mmWave UE: 23 dBm. |
| `enb_tx_power_dbm` | `30.0` | eNB (gNB) transmit power in dBm. |
| `noise_figure_db` | `9.0` | UE receiver noise figure in dB. Higher → worse SNR. |
| `enb_noise_figure_db` | `5.0` | eNB receiver noise figure in dB. |
| `scenario` | `"UMa"` | 3GPP TR 38.901 propagation scenario (see table below). |
| `harq_enabled` | `True` | Enable Hybrid ARQ. Retransmissions combine for improved decoding; adds 1–5 ms per round. |
| `rlc_am_enabled` | `False` | Enable RLC Acknowledged Mode. Higher reliability, additional delay. Disable for latency-sensitive RL. |
| `packet_size_bytes` | `64` | Default probe packet payload in bytes. |
| `step_duration_ms` | `1.0` | ns-3 simulation time per env step (ms). |
| `max_pending_steps` | `500` | Python-side expiry: packets older than this many steps are discarded as lost. |
| `sim_binary` | `""` | Absolute path to `ns3_mmwave_sim`. Auto-detected as `<project_root>/src/ns3_mmwave_sim` when empty. |

**Propagation scenarios (`scenario`):**

| Value | Description |
|---|---|
| `"RMa"` | Rural Macro — low density, long range |
| `"UMa"` | Urban Macro — outdoor city (default) |
| `"UMi-StreetCanyon"` | Urban Micro — dense urban, street level |
| `"InH-OfficeMixed"` | Indoor Hotspot — mixed LOS/NLOS office |
| `"InH-OfficeOpen"` | Indoor Hotspot — open-plan office |

**Choosing `step_duration_ms` for mmWave:**

| Value | Effect |
|---|---|
| 0.5–1 ms | ~1 TTI; single HARQ round per step; realistic per-packet delay variation |
| 2–5 ms | 2–5 HARQ rounds fit per step; most packets arrive same step or are dropped |
| 10+ ms | Very coarse; near-zero delay variation |

---

## NetworkedEnv API

```python
class NetworkedEnv(gymnasium.Wrapper):

    def __init__(
        self,
        env: gymnasium.Env,
        config: NetworkConfig,
        channel_config: Optional[Union[NS3WifiConfig, NS3MmWaveConfig]] = None,
        node_id: str = "agent_0",
    )
```

| Parameter | Description |
|---|---|
| `env` | Base gymnasium environment. Must have a `Box` observation space. |
| `config` | `NetworkConfig`. Controls buffer size and GE channel parameters. |
| `channel_config` | `None` → GE channel. `NS3WifiConfig(...)` → ns-3 WiFi. `NS3MmWaveConfig(...)` → ns-3 5G mmWave. |
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

Both ns-3 backends model full MAC/PHY stacks, so packet size directly affects **transmission time** and **drop probability**. You can override it per step:

```python
# Small control packet
obs, *_ = env.step(action, packet_size=64)

# Large observation payload — longer TX time, higher collision/error probability
obs, *_ = env.step(action, packet_size=2048)

# Use the default from NS3WifiConfig.packet_size_bytes / NS3MmWaveConfig.packet_size_bytes
obs, *_ = env.step(action)
```

> `packet_size` is silently ignored by the GE and Perfect backends.

---

## Building the ns-3 WiFi Binary

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

## Building the ns-3 5G mmWave Binary

The mmWave binary (`ns3_mmwave_sim`) requires **ns-3-mmwave 3.42** — a fork of ns-3 with a full 5G NR/EPC sub-6/mmWave stack. It cannot use the pip-installed ns-3 package.

### Prerequisites (mmWave)

1. **GCC ≥ 10** with C++20 support (`g++ --version`)
2. A built copy of [ns-3-mmwave](https://github.com/nyuwireless-unipd/ns3-mmwave) at `ns3-mmwave/build`:

```bash
git clone https://github.com/nyuwireless-unipd/ns3-mmwave.git
cd ns3-mmwave
./waf configure --build-profile=optimized --disable-python
./waf build
```

The build script looks for the ns-3-mmwave tree at `/home/dianalab/Projects/ns3-mmwave`. Edit the `NS3_MMWAVE_BUILD` variable at the top of `src/build_ns3_mmwave_sim.sh` if your path differs.

Required ns-3-mmwave modules (all built by default): `core`, `network`, `internet`, `point-to-point`, `mobility`, `spectrum`, `mmwave`.

### Build script (mmWave)

```bash
bash src/build_ns3_mmwave_sim.sh
```

The script:
1. Checks that `${NS3_MMWAVE_BUILD}/lib/libns3.42-mmwave-default.so` exists
2. Compiles `src/ns3_mmwave_sim.cc` with `-std=c++20` and all required include/library paths
3. Embeds the library search path with `-Wl,-rpath`
4. Runs a smoke test (`printf 'QUIT\n' | src/ns3_mmwave_sim`)
5. Places the output at `src/ns3_mmwave_sim`

Expected output:

```
=== NetRL ns3 mmWave simulation build ===
  Source : .../src/ns3_mmwave_sim.cc
  ns3    : ns3-mmwave 3.42 at .../ns3-mmwave/build

Compiling with -std=c++20 -O2 ...
Built: .../src/ns3_mmwave_sim

Smoke test (QUIT) ...
Smoke test PASSED
```

### Build options (mmWave)

```bash
bash src/build_ns3_mmwave_sim.sh            # Release build (-O2, default)
bash src/build_ns3_mmwave_sim.sh --release  # Explicit release
bash src/build_ns3_mmwave_sim.sh --debug    # Debug build (-O0 -g)
```

### Manual compilation (mmWave)

```bash
NS3_BUILD=/path/to/ns3-mmwave/build

g++ -std=c++20 -O2 \
    -I${NS3_BUILD}/include \
    -L${NS3_BUILD}/lib \
    -Wl,-rpath,${NS3_BUILD}/lib \
    src/ns3_mmwave_sim.cc \
    -lns3.42-core-default \
    -lns3.42-network-default \
    -lns3.42-internet-default \
    -lns3.42-point-to-point-default \
    -lns3.42-mobility-default \
    -lns3.42-spectrum-default \
    -lns3.42-mmwave-default \
    -o src/ns3_mmwave_sim
```

> Verify the exact library filenames with:
> ```bash
> ls ${NS3_BUILD}/lib/libns3.42-mmwave*
> ```

### Verification (mmWave)

```bash
printf 'QUIT\n' | src/ns3_mmwave_sim
# Expected single line of output: READY
```

> **Note:** The first `reset()` call after process start takes up to 60 s on slow machines while ns-3 loads shared libraries and the EPC performs its 500 ms warm-up. Subsequent resets are faster.

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

**`RuntimeError: ns3_mmwave_sim binary not found`**

The mmWave binary has not been compiled. Run:
```bash
bash src/build_ns3_mmwave_sim.sh
```

If the script cannot find the ns-3-mmwave build tree, edit `NS3_MMWAVE_BUILD` at the top of the script.

**`RuntimeError: NS3MmWaveChannel: subprocess stdout closed`** (crash, return code -6)

The `ns3_mmwave_sim` process aborted. Common causes:
- Wrong attribute names (ns-3-mmwave version mismatch) — rebuild with the correct source
- Binary was compiled with `-std=c++17` — recompile with `-std=c++20`

**First `reset()` takes a long time (mmWave)**

The 5G EPC performs a 500 ms simulated warm-up during which the UE attaches and a default bearer is established. On the first call, ns-3 also loads mmWave shared libraries — expect up to 60 s total. Subsequent resets are significantly faster.

**Build script fails with a `glib-2.0` or CMake path error**

This is a known issue with some pip ns-3 installations where the CMake config references non-existent system paths. The build script bypasses CMake and calls `g++` directly to avoid this. Ensure you are using the current `src/build_ns3_sim.sh`.
