import gymnasium as gym

from netrl import NetworkConfig, NetworkedEnv, NS3LenaConfig


env = NetworkedEnv(
    gym.make("CartPole-v1"),
    NetworkConfig(buffer_size=10, seed=42),
    channel_config=NS3LenaConfig(
        distance_m=80.0,
        frequency_ghz=28.0,
        bandwidth_ghz=0.2,
        scenario="UMa",
        numerology=3,
        step_duration_ms=20.0,
        packet_size_bytes=128,
    ),
)

obs, info = env.reset()
print(obs.keys())

for i in range(100):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if i % 10 == 0:
        print(obs["recv_mask"])
    if term or trunc:
        obs, info = env.reset()

    
