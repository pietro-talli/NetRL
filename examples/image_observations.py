from netrl import NetworkedEnv, NetworkConfig
from netrl.ns3_wifi_config import NS3WifiConfig
import gymnasium as gym
from netrl.image_env_wrapper import ImageEnvWrapper
import numpy as np

env = gym.make("CartPole-v1", render_mode = "rgb_array")
config = NetworkConfig(
    buffer_size=10,  # observation window length
    seed=42,
)

env = ImageEnvWrapper(env, height=64, width=64, channels=3)

net_env = NetworkedEnv(env, 
                       config, 
                       channel_config=NS3WifiConfig(step_duration_ms=20.0, 
                                                    distance_m=45.0,
                                                    tx_power_dbm=20.0,
                                                    loss_exponent=3.0))

obs, info = net_env.reset()
print(obs.keys())  # dict_keys(['observations', 'recv_mask'])
print(obs["observations"].shape)  # (10, 4)
print(obs["recv_mask"].shape)     # (10,)
obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
print(info["channel_info"]["state"])        # "GOOD" or "BAD"
print(info["arrived_this_step"])            # True / False

size = 64*64*3 # packet size in bytes for the ns-3 channel model; adjust as needed

import time
start_time = time.time()

for i in range(1000):
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample(), 
                                                  packet_size=size)
    if term or trunc:
        env.reset()

print(f"FPS: {1000 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])
print(obs["observations"].shape)