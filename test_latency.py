from netrl import NS3WiFiChannelFast, NetworkConfig
import numpy as np

channel = NS3WiFiChannelFast(NetworkConfig(),
                             distance_m=60.0,
                             step_duration_ms=5.0)

obs=np.zeros(10000)
channel.reset()  # Reset the channel to initialize it
delays = []
num_tx = 0

for step in range(1000):
    if channel.get_channel_info()["pending_tx_count"] > 10:
        continue
    else:
        channel.transmit(obs, step, 25000)
        res = channel.flush(step)
        for transmission_step, obs in res:
            delay = step - transmission_step
            delays.append(delay)
            num_tx += 1

print("Delays:", np.mean(delays), np.std(delays), num_tx)

import time

import gymnasium as gym
import numpy as np

from netrl import CentralNode, NetworkConfig
from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory

# ---------------------------------------------------------------------------
# 1. Configure the multi-UE WiFi network
# ---------------------------------------------------------------------------
N_UES = 5
node_ids = [f"ue_{i}" for i in range(N_UES)]

ns3_cfg = NS3WifiMultiUEConfig(
    n_ues=N_UES,
    distances_m=[30.0] * N_UES,   # All UEs at 30 m → uniform losses / retries
    step_duration_ms=5.0,
    tx_power_dbm=20.0,
    loss_exponent=3.4,
    max_retries=7,
    packet_size_bytes=1000,
)

# make_multi_ue_wifi_factory() starts the subprocess immediately and returns
# a factory that hands out one NS3WifiUEChannel per CentralNode registration.
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

central.reset()  # Reset to initialize channels
delays = []

start_time = time.time()

for step in range(1000):
    for node_id in node_ids:
        if central.get_channel_info(node_id)["pending_count"] < 10:
            obs = np.random.rand(*OBS_SHAPE).astype(OBS_DTYPE)
            central.receive_from(node_id, obs, step, 2000)

    for node_id in node_ids:
        res = central._channels[node_id].flush(step=step)
        for transmission_step, obs in res:
            delay = step - transmission_step
            delays.append(delay)

print(f"FPS: {1000 / (time.time() - start_time):.2f}")

print("Delays:", np.mean(delays), np.std(delays), len(delays))