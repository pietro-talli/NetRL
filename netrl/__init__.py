"""
NetRL — Networked Reinforcement Learning Simulation Platform
============================================================
A gymnasium wrapper that simulates a noisy communication channel between
an RL agent and its environment.  Each step() call transmits the raw
observation through a C++ Gilbert-Elliott channel (loss + delay) to a
central node.  The agent receives the full observation buffer instead of
the raw observation.

Quick start
-----------
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
    base_env = gym.make("CartPole-v1")
    env = NetworkedEnv(base_env, config)

    obs, info = env.reset()
    # obs["observations"].shape == (10, 4)
    # obs["recv_mask"].shape    == (10,)

    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(info["channel_info"]["state"])        # "GOOD" or "BAD"
    print(info["arrived_this_step"])            # True / False

Using a baseline (no C++ required)
-----------------------------------
    from netrl import PerfectChannel
    env = NetworkedEnv(base_env, config, channel_factory=PerfectChannel)

Future ns3 backend
------------------
    class NS3Channel(CommChannel):
        ...  # implement transmit / flush / reset / get_channel_info

    env = NetworkedEnv(base_env, config,
                       channel_factory=lambda cfg: NS3Channel(cfg, ns3_handle))
"""

from netrl.observation_buffer import ObservationBuffer
from netrl.node import Node
from netrl.network_config import NetworkConfig
from netrl.comm_channel import CommChannel, GEChannel, PerfectChannel
from netrl.central_node import CentralNode
from netrl.networked_env import NetworkedEnv

__version__ = "0.1.0"

__all__ = [
    "ObservationBuffer",
    "Node",
    "NetworkConfig",
    "CommChannel",
    "GEChannel",
    "PerfectChannel",
    "CentralNode",
    "NetworkedEnv",
]
