Quick Start
===========

This page gets you from installation to a running networked CartPole in five
minutes, using the built-in Gilbert–Elliott channel (no ns-3 required).

.. contents:: On this page
   :local:
   :depth: 2

Single observer
---------------

Wrap any Gymnasium environment with :class:`~netrl.NetworkedEnv`:

.. code-block:: python

   import gymnasium as gym
   from netrl import NetworkedEnv, NetworkConfig

   config = NetworkConfig(
       p_gb=0.10,       # Good → Bad transition probability per step
       p_bg=0.30,       # Bad  → Good transition probability per step
       loss_good=0.01,  # Packet loss in Good state
       loss_bad=0.20,   # Packet loss in Bad state
       delay_steps=2,   # Propagation delay (integer steps)
       buffer_size=10,  # Observation window length
       seed=42,
   )
   env = NetworkedEnv(gym.make("CartPole-v1"), config)

   obs, info = env.reset()
   print(obs["observations"].shape)   # (10, 4) — 10-step window, 4-dim obs
   print(obs["recv_mask"].shape)      # (10,)   — True where packet arrived

   obs, reward, term, trunc, info = env.step(env.action_space.sample())
   print(info["channel_info"]["state"])   # "GOOD" or "BAD"
   print(info["arrived_this_step"])       # True / False

The agent receives a **sliding-window buffer** of the last ``buffer_size``
observations.  Slots where packets were lost or delayed are zero-filled;
``recv_mask`` marks which slots actually received data.

Multiple observers (multi-view)
--------------------------------

Use :class:`~netrl.MultiViewNetworkedEnv` when multiple sensors each transmit
their own (possibly different-shaped) observations to a central node.

.. code-block:: python

   import numpy as np
   import gymnasium as gym
   from netrl import NetworkConfig, MultiViewNetworkedEnv
   from netrl.channels.comm_channel import GEChannel
   from netrl.utils.multi_view_model import MultiViewModel

   # --- Define what each sensor observes ---
   class MySensors(MultiViewModel):
       def observe(self, env, state):
           # state is the raw gym observation; each sensor produces its own
           return {
               "lidar":  state[:2].astype(np.float32),       # 2-dim
               "camera": np.random.randn(8).astype(np.float32),  # 8-dim
           }

   mv_model = MySensors(
       observer_ids=["lidar", "camera"],
       obs_shapes=[(2,), (8,)],
       obs_dtypes=[np.float32, np.float32],
   )

   env = MultiViewNetworkedEnv(
       gym.make("CartPole-v1"),
       NetworkConfig(buffer_size=8, loss_bad=0.3),
       observer_ids=["lidar", "camera"],
       multi_view_model=mv_model,
       channel_factory=GEChannel,
   )

   obs, info = env.reset()
   # obs["lidar"]["observations"].shape  == (8, 2)
   # obs["camera"]["observations"].shape == (8, 8)

   # Per-step control: only transmit lidar this step, with 64 bytes
   obs, r, term, trunc, info = env.step(
       env.action_space.sample(),
       transmit_mask={"lidar": True, "camera": False},
       packet_sizes={"lidar": 64},
   )
   print(info["transmitted_this_step"])   # {"lidar": True, "camera": False}
   print(info["arrived_this_step"])       # {"lidar": True/False, "camera": False}

Switching to a realistic ns-3 WiFi channel
-------------------------------------------

Build the binary once, then pass an :class:`~netrl.NS3WifiConfig`:

.. code-block:: bash

   bash src/build_ns3_sim.sh

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3WifiConfig

   env = NetworkedEnv(
       gym.make("CartPole-v1"),
       NetworkConfig(buffer_size=10),
       channel_config=NS3WifiConfig(
           distance_m=30.0,
           step_duration_ms=2.0,
           tx_power_dbm=20.0,
           loss_exponent=3.0,
       ),
   )

The ns-3 process runs continuously alongside your Python code; the simulation
state (MAC buffers, backoff counters) persists across steps.

Next steps
----------

* :doc:`guides/single_observer` — full NetworkedEnv usage guide
* :doc:`guides/multi_view` — complete MultiViewNetworkedEnv guide
* :doc:`guides/channels` — choosing and configuring channel backends
* :doc:`api/index` — full API reference
