from netrl import NetworkedEnv, NetworkConfig
from netrl.ns3_wifi_config import NS3WifiConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = NetworkConfig(
    p_gb=0.1,        # Good -> Bad transition probability
    p_bg=0.3,        # Bad  -> Good transition probability
    loss_good=0.01,  # packet loss in Good state
    loss_bad=0.20,   # packet loss in Bad state
    delay_steps=3,   # one-way propagation delay (steps)
    buffer_size=10,  # observation window length
    seed=42,
)

net_env = NetworkedEnv(env, config, channel_config=NS3WifiConfig(step_duration_ms=10.0, distance_m=52.0))
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
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
    if term or trunc:
        env.reset()

print(f"FPS: {100 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])


for i in range(1000):
    obs, reward, term, trunc, info = net_env.step(net_env.action_space.sample())
    if term or trunc:
        env.reset()

print(f"FPS: {1000 / (time.time() - start_time):.2f}")

print(obs["recv_mask"])