from netrl import NetworkedEnv, NetworkConfig
from netrl.ns3_wifi_config import NS3WifiConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = NetworkConfig(
    buffer_size=10,  # observation window length
    seed=42,
)

net_env = NetworkedEnv(env, 
                       config, 
                       channel_config=NS3WifiConfig(step_duration_ms=10.0, 
                                                    distance_m=30.0,
                                                    tx_power_dbm=20.0,
                                                    packet_size_bytes=10000))
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
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
    if term or trunc:
        env.reset()

print(f"FPS: {1000 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])
print(obs["observations"])