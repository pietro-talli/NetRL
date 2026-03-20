# NetRL — Networked Reinforcement Learning Simulation Platform

NetRL wraps any [Gymnasium](https://gymnasium.farama.org/) environment and simulates a **noisy communication channel** between the agent and the environment. At every `step()`, the raw observation is transmitted through a configurable channel backend (loss, delay, retransmissions). The agent receives a **sliding-window buffer** of past observations together with a boolean mask indicating which slots actually arrived.

Full documentation: https://netrl.readthedocs.io/en/latest/index.html

Four channel backends are available:

| Backend | Model | Config class | Requires |
|---|---|---|---|
| **Gilbert-Elliott** (default) | Two-state Markov chain with configurable loss per state and fixed delay | `NetworkConfig` | Built-in |
| **ns-3 802.11a WiFi (fast)** ⚡ | Full MAC/PHY simulation via ns-3 — same physics as below but compiled as a Python C++ extension (pybind11), **no subprocess overhead** | `NS3WiFiChannelFastConfig` | `pip install ns3` — built automatically |
| **ns-3 802.11a WiFi** | Full MAC/PHY simulation via ns-3 — CSMA/CA, retransmissions, path-loss | `NS3WifiConfig` | `pip install ns3` + manual binary build |
| **ns-3 5G mmWave** | Full EPC/NR simulation via ns-3-mmwave — 3GPP TR 38.901 path-loss, HARQ, RLC | `NS3MmWaveConfig` | ns-3-mmwave source build |
| **ns-3 5G-LENA NR** | Full NR + EPC simulation via 5G-LENA (contrib/nr) — 3GPP channel, beamforming, numerology | `NS3LenaConfig` | 5G-LENA source build |

> **Recommendation:** Use `NS3WiFiChannelFastConfig` for all 802.11a WiFi experiments. It provides identical physical simulation to `NS3WifiConfig` but runs **2-3× faster** by eliminating subprocess IPC overhead — and it is compiled automatically when you run `pip install -e .`.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Python package, GE channel, and fast WiFi channel](#1-python-package-ge-channel-and-fast-wifi-channel)
   - [ns-3 WiFi binary (subprocess version)](#2-ns-3-wifi-binary-subprocess-version)
   - [ns-3 5G mmWave binary](#3-ns-3-5g-mmwave-binary)
    - [ns-3 5G-LENA binary](#4-ns-3-5g-lena-binary)
3. [Quick Start](#quick-start)
   - [Gilbert-Elliott channel](#gilbert-elliott-channel)
   - [ns-3 WiFi channel (fast — recommended)](#ns-3-wifi-channel-fast--recommended)
   - [ns-3 WiFi channel (subprocess)](#ns-3-wifi-channel-subprocess)
   - [ns-3 5G mmWave channel](#ns-3-5g-mmwave-channel)
    - [ns-3 5G-LENA channel](#ns-3-5g-lena-channel)
4. [Observation Space](#observation-space)

---

## Requirements

- Python ≥ 3.10
- GCC ≥ 10 or Clang ≥ 11
  - C++20 for the pybind11 extensions (`netcomm` GE channel and `netrl_ext` fast WiFi channel)
  - **C++20** for the ns-3 mmWave binary (ns-3-mmwave 3.42 uses `std::remove_cvref_t`)
- For the fast WiFi channel and the subprocess WiFi channel:
  - `pip install ns3` (ns-3 ≥ 3.43, headers + shared libs) — the **fast WiFi channel is built automatically** during `pip install -e .`
- For the 5G mmWave channel:
  - A compiled **ns-3-mmwave** source build at `/path/to/ns3-mmwave/build` (ns-3 3.42)

---

## Installation

### 1. Python package, GE channel, and fast WiFi channel

The Gilbert-Elliott channel (`netcomm`) and the fast ns-3 WiFi channel (`_netrl_ext`) are both compiled as C++ pybind11 extensions. Install everything with a single command:

```bash
pip install -e .
```

This automatically:
1. Installs all Python dependencies (including `ns3 ≥ 3.44`)
2. Compiles `netcomm` — the Gilbert-Elliott C++ backend
3. Detects the pip-installed `ns3` library and **compiles `_netrl_ext`** — the fast WiFi pybind11 binding

The fast WiFi channel (`NS3WiFiChannelFastConfig`) is then immediately available — no extra build step required.

```python
from netrl import NetworkedEnv, NetworkConfig, NS3WiFiChannelFastConfig
```

### 2. ns-3 WiFi binary (subprocess version)

The subprocess ns-3 backend runs the simulation in a **separate process** and communicates via stdin/stdout pipes. Compile the binary once before use:

```bash
bash src/build_ns3_sim.sh
```

The script auto-detects your ns-3 installation, compiles `src/ns3_wifi_sim.cc`, and writes the binary to `src/ns3_wifi_sim`.

### 3. ns-3 5G mmWave binary

The mmWave backend requires a separate binary built against [ns-3-mmwave](https://github.com/nyuwireless-unipd/ns3-mmwave) (ns-3 3.42). Compile it once before use:

```bash
bash src/build_ns3_mmwave_sim.sh --ns3-mmwave-dir /path/to/ns3-mmwave
```

The script expects ns-3-mmwave to be built at `/home/dianalab/Projects/ns3-mmwave/build` (edit the `NS3_MMWAVE_BUILD` variable at the top of the script to change the path).

### 4. ns-3 5G-LENA binary

The 5G-LENA backend requires a separate binary built against your local 5G-LENA tree (contrib/nr). Compile it once before use:

```bash
bash src/build_ns3_lena_sim.sh --ns3-lena-dir /path/to/5g-lena/ns-3-dev
```

By default, the script expects 5G-LENA at `/home/dianalab/Projects/5g-lena/ns-3-dev`.

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
from netrl import NetworkedEnv, NetworkConfig, NS3WiFiChannelFastConfig

env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10, seed=42),
    channel_config=NS3WiFiChannelFastConfig(
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

> **No subprocess, no binary to build.** The `netrl_ext` extension is compiled automatically during `pip install -e .` and linked directly into the Python process.  Performance is **2-3× faster** than the subprocess variant, with lower memory usage and instant startup.

### ns-3 WiFi channel (subprocess)

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

### ns-3 5G-LENA channel

```python
import gymnasium as gym
from netrl import NetworkedEnv, NetworkConfig, NS3LenaConfig

env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10),
    channel_config=NS3LenaConfig(
        distance_m=80.0,
        frequency_ghz=28.0,
        bandwidth_ghz=0.1,
        scenario="UMa",
        numerology=3,
        step_duration_ms=2.0,
    ),
)

obs, info = env.reset()
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])   # "NS3_LENA"
    if term or trunc:
        obs, info = env.reset()
```

> The 5G-LENA simulation is persistent across steps and only rebuilt on `env.reset()`. It uses the same NetRL subprocess protocol as the WiFi and mmWave backends.

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

**ns-3 fast WiFi `channel_info` keys:** `state` (`"NS3_WIFI"`), `distance_m`, `step_duration_ms`, `tx_power_dbm`, `loss_exponent`, `max_retries`, `packet_size_bytes`, `pending_tx_count`, `pending_rx_count`

**ns-3 WiFi `channel_info` keys:** `state` (`"NS3_WIFI"`), `pending_count`, `arrived_buffered`, `distance_m`, `step_duration_ms`, `tx_power_dbm`, `loss_exponent`, `max_retries`

**ns-3 mmWave `channel_info` keys:** `state` (`"NS3_MMWAVE"`), `pending_count`, `arrived_buffered`, `distance_m`, `frequency_ghz`, `bandwidth_ghz`, `tx_power_dbm`, `enb_tx_power_dbm`, `noise_figure_db`, `scenario`, `harq_enabled`, `rlc_am_enabled`, `step_duration_ms`

