"""
cartpole_wifi_multi_ue.py
=========================
Example: multiple UEs sending CartPole observations over a shared
802.11a infrastructure WiFi network to a single AP (central node).

Topology
--------
   UE 0 (10 m) ──┐
   UE 1 (30 m) ──┤── 802.11a BSS ──> AP (CentralNode)
   UE 2 (60 m) ──┘

All three UEs observe the CartPole state at every step and transmit it
independently.  They share the same wireless medium; CSMA/CA contention
is simulated inside ns-3.  The AP buffers arriving observations per UE.

Build the binary before running:
    bash src/build_ns3_multi_ue_sim.sh
"""

import time

import gymnasium as gym
import numpy as np

from netrl import CentralNode, NetworkConfig
from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory

# ---------------------------------------------------------------------------
# 1. Configure the multi-UE WiFi network
# ---------------------------------------------------------------------------
N_UES = 3
node_ids = [f"ue_{i}" for i in range(N_UES)]

ns3_cfg = NS3WifiMultiUEConfig(
    n_ues=N_UES,
    distances_m=[10.0, 30.0, 50.0],   # UE 2 is far → more losses / retries
    step_duration_ms=20.0,
    tx_power_dbm=20.0,
    loss_exponent=3.0,
    max_retries=7,
    packet_size_bytes=512,
)

# make_multi_ue_wifi_factory() starts the subprocess immediately and returns
# a factory that hands out one NS3WifiUEChannel per CentralNode registration.
print("Starting ns-3 multi-UE simulator (association warm-up ~500 ms)…")
factory = make_multi_ue_wifi_factory(ns3_cfg)

# ---------------------------------------------------------------------------
# 2. Set up CentralNode with one buffer per UE
# ---------------------------------------------------------------------------
net_config = NetworkConfig(buffer_size=10, seed=42)

# The CartPole observation space is Box(4,) float32
OBS_SHAPE = (4,)
OBS_DTYPE = np.float32

central = CentralNode(
    node_ids=node_ids,
    obs_shape=OBS_SHAPE,
    obs_dtype=OBS_DTYPE,
    config=net_config,
    channel_factory=factory,
)

# ---------------------------------------------------------------------------
# 3. Wrap CartPole
# ---------------------------------------------------------------------------
env = gym.make("CartPole-v1")
raw_obs, _ = env.reset()
central.reset()

print(f"\nRunning {N_UES} UEs over shared 802.11a WiFi for 500 steps…\n")

step = 0
n_steps = 1000
arrived_counts = {nid: 0 for nid in node_ids}

t0 = time.time()

for _ in range(n_steps):
    # All UEs transmit the same CartPole observation in this step.
    # In a real multi-agent scenario each UE would have its own observation.
    for nid in node_ids:
        central.receive_from(nid, raw_obs, step, packet_size=ns3_cfg.packet_size_bytes)

    # Flush all UE channels (single subprocess call, cached for each UE)
    arrived_map = central.flush_and_update(step)

    for nid, obs in arrived_map.items():
        if obs is not None:
            arrived_counts[nid] += 1

    # Step the environment
    action = env.action_space.sample()
    raw_obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        raw_obs, _ = env.reset()

    step += 1

elapsed = time.time() - t0
fps = n_steps / elapsed

# ---------------------------------------------------------------------------
# 4. Report results
# ---------------------------------------------------------------------------
print(f"Completed {n_steps} steps in {elapsed:.1f} s  ({fps:.1f} steps/s)\n")
print("Per-UE packet delivery statistics:")
for nid in node_ids:
    pdr = arrived_counts[nid] / n_steps * 100
    info = central.get_channel_info(nid)
    print(
        f"  {nid}  dist={info['distance_m']:5.1f} m  "
        f"arrived={arrived_counts[nid]}/{n_steps}  PDR={pdr:.1f}%"
    )

# Show the final observation buffer for each UE
print("\nFinal observation buffers (most-recent entry last):")
all_bufs = central.get_all_buffers()
for nid, (obs_buf, recv_mask) in all_bufs.items():
    print(f"  {nid}  recv_mask={recv_mask.astype(int).tolist()}")
    print(f"         last obs ={obs_buf[-1].round(3).tolist()}")

env.close()
