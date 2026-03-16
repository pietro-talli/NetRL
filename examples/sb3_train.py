# pip install stable-baselines3 
# (Required for this example, not a dependency of netrl)
from stable_baselines3 import PPO 

from netrl import NS3WifiConfig, NetworkedEnv, NetworkConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = NetworkConfig(
    buffer_size=10)  # observation window length
ns_config = NS3WifiConfig(
    step_duration_ms=20.0,
    distance_m=50.0,
    tx_power_dbm=20.0,
    loss_exponent=3.0,
    packet_size_bytes=256)

net_env = NetworkedEnv(env, config, channel_config=ns_config)

model = PPO("MultiInputPolicy", net_env, verbose=1)
model.learn(total_timesteps=100000)