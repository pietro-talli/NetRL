"""
cartpole_multi_view.py
======================
Demonstrates MultiViewNetworkedEnv with three observers watching CartPole.

Each observer transmits its copy of the state through its own channel.
This example shows:
  - Basic setup with the Gilbert–Elliott channel (no ns-3 build needed)
  - Per-step transmission masking (selective observer activation)
  - Per-step variable packet sizes (variable-rate encoding)
  - Reading per-observer observation buffers and statistics

To run with a shared 802.11a WiFi channel instead of GE, uncomment the
ns-3 block at the bottom of the configuration section.

Usage:
    python examples/cartpole_multi_view.py
"""

import gymnasium as gym
import numpy as np

from netrl import NetworkConfig, MultiViewNetworkedEnv
from netrl.channels.comm_channel import GEChannel
from netrl.utils.multi_view_model import MultiViewModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OBSERVER_IDS = ["sensor_A", "sensor_B", "sensor_C"]
BUFFER_SIZE  = 8

config = NetworkConfig(
    p_gb=0.05,         # Good → Bad transition
    p_bg=0.40,         # Bad  → Good transition
    loss_good=0.02,    # 2 % loss in Good state
    loss_bad=0.30,     # 30 % loss in Bad state
    delay_steps=0,     # 2-step propagation delay
    buffer_size=BUFFER_SIZE,
    seed=42,
)

# --- GE channel: each observer gets an independent channel instance ---
channel_factory = GEChannel

# --- Uncomment to use a shared 802.11a WiFi BSS instead (requires build) ---
from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory
channel_factory = make_multi_ue_wifi_factory(
    NS3WifiMultiUEConfig(
        n_ues=len(OBSERVER_IDS),
        distances_m=[10.0, 30.0, 40.0],
        step_duration_ms=2.0,
    )
)

# ---------------------------------------------------------------------------
# Implement the observation model
# ---------------------------------------------------------------------------
class My_MV(MultiViewModel):
    def __init__(self, **args):
        super().__init__(**args)

    def observe(self, env, state):
        return {self.observer_ids[0]: np.random.randn(5), 
                self.observer_ids[1]: np.random.randn(3), 
                self.observer_ids[2]: np.random.randn(10)}

multi_view_model = My_MV(observer_ids=OBSERVER_IDS,
                         obs_shapes = [(5,), (3,), (10,)],
                         obs_dtypes = [np.float32, np.float32, np.float32])

# ---------------------------------------------------------------------------
# Build the environment
# ---------------------------------------------------------------------------
env = MultiViewNetworkedEnv(
    gym.make("CartPole-v1"),
    config,
    observer_ids=OBSERVER_IDS,
    multi_view_model=multi_view_model,
    channel_factory=channel_factory,
)

print("Observation space:")
for oid in env.observer_ids:
    obs_sp  = env.observation_space[oid]["observations"]
    mask_sp = env.observation_space[oid]["recv_mask"]
    print(f"  {oid}  observations={obs_sp.shape}  recv_mask={mask_sp.n}")

# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------
obs, info = env.reset()
assert set(obs.keys()) == set(OBSERVER_IDS)

N_STEPS = 300
arrived  = {o: 0 for o in OBSERVER_IDS}
tx_count = {o: 0 for o in OBSERVER_IDS}

print(f"\nRunning {N_STEPS} steps …\n")

for step_idx in range(N_STEPS):
    action = env.action_space.sample()

    # ------------------------------------------------------------------
    # Transmission policy examples
    # ------------------------------------------------------------------
    if step_idx % 3 == 0:
        # All three sensors transmit; give sensor_A a larger packet
        obs, r, term, trunc, info = env.step(
            action,
            packet_sizes={"sensor_A": 256, "sensor_B": 64, "sensor_C": 64},
        )

    elif step_idx % 3 == 1:
        # Only sensor_A and sensor_B transmit this step (duty cycling)
        obs, r, term, trunc, info = env.step(
            action,
            transmit_mask={"sensor_A": True, "sensor_B": True, "sensor_C": False},
        )

    else:
        # Only sensor_C transmits, with a small packet
        obs, r, term, trunc, info = env.step(
            action,
            transmit_mask={"sensor_A": False, "sensor_B": False, "sensor_C": True},
            packet_sizes={"sensor_C": 32},
        )

    # Accumulate statistics
    for o in OBSERVER_IDS:
        if info["arrived_this_step"][o]:
            arrived[o] += 1
        if info["transmitted_this_step"][o]:
            tx_count[o] += 1

    if term or trunc:
        obs, _ = env.reset()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("Per-observer delivery statistics:")
print(f"{'Observer':<12}  {'TX':>6}  {'RX':>6}  {'PDR':>7}  {'Channel state':>14}")
for o in OBSERVER_IDS:
    pdr   = arrived[o] / tx_count[o] * 100 if tx_count[o] else 0.0
    state = info["channel_info"][o].get("state", "—")
    print(f"  {o:<10}  {tx_count[o]:>6}  {arrived[o]:>6}  {pdr:>6.1f}%  {state:>14}")

print("\nFinal observation buffers (recv_mask only):")
for o in OBSERVER_IDS:
    print(f"  {o}: recv_mask = {obs[o]['recv_mask'].tolist()}")

print([obs[o]["observations"].shape for o in OBSERVER_IDS])

env.close()
