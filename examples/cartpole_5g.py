from netrl import NetworkedEnv, NetworkConfig
from netrl.channels.ns3_mmwave_config import NS3MmWaveConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = NetworkConfig(
    buffer_size=10,  # observation window length
    seed=42,
)

net_env = NetworkedEnv(env, 
                       config, 
                       channel_config=NS3MmWaveConfig(
                           distance_m=400,
                           frequency_ghz=6.0,
                           step_duration_ms=20.0))

obs, info = net_env.reset()
print(obs.keys())  # dict_keys(['observations', 'recv_mask'])
print(obs["observations"].shape)  # (10, 4)
print(obs["recv_mask"].shape)     # (10,)
obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
print(info["channel_info"]["state"])        # "GOOD" or "BAD"
print(info["arrived_this_step"])            # True / False

import time
start_time = time.time()

for i in range(100):
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample(), 
                                                  packet_size=256)
    if term or trunc:
        env.reset()

print(f"FPS: {100 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])
print(obs["observations"])