import gymnasium as gym

from netrl import NetworkConfig, NetworkedEnv, NS3LenaConfig


net_env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10, seed=42),
    channel_config=NS3LenaConfig(
        distance_m=400.0,
        frequency_ghz=6.0,
        bandwidth_ghz=0.2,
        scenario="UMa",
        numerology=5,
        step_duration_ms=20.0,
        packet_size_bytes=128,
    ),
)

obs, info = net_env.reset()
print(obs.keys())  # dict_keys(['observations', 'recv_mask'])
print(obs["observations"].shape)  # (10, 4)
print(obs["recv_mask"].shape)     # (10,)
obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
print(info["channel_info"]["state"])        # "GOOD" or "BAD"
print(info["arrived_this_step"])            # True / False

import time
start_time = time.time()

for i in range(1000):
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample(), 
                                                  packet_size=256)
    if term or trunc:
        net_env.reset()

print(f"FPS: {1000 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])
print(obs["observations"])