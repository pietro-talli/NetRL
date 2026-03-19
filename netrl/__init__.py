"""
NetRL — Networked Reinforcement Learning Simulation Platform
============================================================
A gymnasium wrapper that simulates a noisy communication channel between
an RL agent and its environment.  Each step() call transmits the raw
observation through a configurable channel backend (loss + delay) to a
central node.  The agent receives the full observation buffer instead of
the raw observation.

Quick start — Gilbert-Elliott channel (default)
------------------------------------------------
    import gymnasium as gym
    from netrl import NetworkedEnv, NetworkConfig

    config = NetworkConfig(
        p_gb=0.1,        # Good -> Bad transition probability
        p_bg=0.3,        # Bad  -> Good transition probability
        loss_good=0.01,  # packet loss in Good state
        loss_bad=0.20,   # packet loss in Bad state
        delay_steps=3,   # one-way propagation delay (steps)
        buffer_size=10,  # observation window length
        seed=42,
    )
    env = NetworkedEnv(gym.make("CartPole-v1"), config)

    obs, info = env.reset()
    # obs["observations"].shape == (10, 4)
    # obs["recv_mask"].shape    == (10,)

    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])        # "GOOD" or "BAD"
    print(info["arrived_this_step"])            # True / False

ns3 WiFi backend (realistic channel)
--------------------------------------
    Build the binary once:
        bash src/build_ns3_sim.sh

    Then select it via channel_config:
        from netrl import NetworkedEnv, NetworkConfig, NS3WifiConfig

        env = NetworkedEnv(
            gym.make("CartPole-v1"),
            NetworkConfig(buffer_size=10),
            channel_config=NS3WifiConfig(distance_m=15.0, step_duration_ms=2.0),
        )

    The simulation persists across steps and only resets on env.reset().

ns3 WiFi backend (fast pybind11 binding - 15-20x faster!)
-----------------------------------------------------------
    Build the extension once:
        bash build_pybind11.sh

    Then select it via channel_config:
        from netrl import NetworkedEnv, NetworkConfig, NS3WiFiChannelFastConfig

        env = NetworkedEnv(
            gym.make("CartPole-v1"),
            NetworkConfig(buffer_size=10),
            channel_config=NS3WiFiChannelFastConfig(distance_m=15.0, step_duration_ms=2.0),
        )

    This uses direct C++ binding with no subprocess overhead!
"""

from netrl.utils.observation_buffer import ObservationBuffer
from netrl.channels.network_config import NetworkConfig
from netrl.channels.comm_channel import CommChannel, GEChannel, PerfectChannel
from netrl.central_node import CentralNode
from netrl.networked_env import NetworkedEnv
from netrl.multi_view_networked_env import MultiViewNetworkedEnv
from netrl.channels.ns3_wifi_config import NS3WifiConfig
from netrl.channels.ns3_channel import NS3WifiChannel
from netrl.channels.ns3_wifi_fast_config import NS3WiFiChannelFastConfig
from netrl.channels.ns3_channel_fast import NS3WiFiChannelFast
from netrl.channels.ns3_mmwave_config import NS3MmWaveConfig
from netrl.channels.ns3_mmwave_channel import NS3MmWaveChannel
from netrl.channels.ns3_lena_config import NS3LenaConfig
from netrl.channels.ns3_lena_channel import NS3LenaChannel
from netrl.channels.ns3_wifi_multi_ue_config import NS3WifiMultiUEConfig
from netrl.channels.ns3_multi_ue_channel import (
    NS3WifiMultiUEBackend,
    NS3WifiUEChannel,
    make_multi_ue_wifi_factory,
)
from netrl.utils.multi_view_model import MultiViewModel

__version__ = "0.2.0"

__all__ = [
    "ObservationBuffer",
    "NetworkConfig",
    "CommChannel",
    "GEChannel",
    "PerfectChannel",
    "CentralNode",
    "NetworkedEnv",
    "MultiViewNetworkedEnv",
    "NS3WifiConfig",
    "NS3WifiChannel",
    "NS3WiFiChannelFastConfig",
    "NS3WiFiChannelFast",
    "NS3MmWaveConfig",
    "NS3MmWaveChannel",
    "NS3LenaConfig",
    "NS3LenaChannel",
    # Multi-UE WiFi
    "NS3WifiMultiUEConfig",
    "NS3WifiMultiUEBackend",
    "NS3WifiUEChannel",
    "make_multi_ue_wifi_factory",
    "MultiViewModel",
]
