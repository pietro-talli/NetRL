from netrl import NetworkedEnv, NetworkConfig
from netrl.channels.ns3_wifi_config import NS3WifiConfig
import gymnasium as gym

from gymnasium.vector import AsyncVectorEnv

envs = [gym.make("CartPole-v1") for _ in range(4)]
config = NetworkConfig(
    buffer_size=10,  # observation window length
    seed=42,
)

def env_fn(i):
    return NetworkedEnv(envs[i], 
                        config, 
                        channel_config=NS3WifiConfig(step_duration_ms=20.0, 
                                                     distance_m=50.0,
                                                     tx_power_dbm=20.0,
                                                     loss_exponent=3.0))

envs = AsyncVectorEnv([lambda i=i: env_fn(i) for i in range(4)])

envs.reset()
obs, reward, ter, trun, info = envs.step(envs.action_space.sample())

print(obs["observations"].shape)  # (4, 10, 4)
