CentralNode
===========

:class:`~netrl.CentralNode` owns one :class:`~netrl.CommChannel` and one
:class:`~netrl.ObservationBuffer` per registered node.  It is the aggregation
layer used internally by both environment wrappers, and can also be used
directly for custom multi-agent pipelines.

.. autoclass:: netrl.CentralNode
   :members:
   :special-members: __init__
   :show-inheritance:

Direct usage example
--------------------

.. code-block:: python

   import numpy as np
   import gymnasium as gym
   from netrl import CentralNode, NetworkConfig
   from netrl.channels.comm_channel import GEChannel

   central = CentralNode(
       node_ids=["agent_0", "agent_1"],
       obs_shape=(4,),          # single tuple → same shape for all nodes
       obs_dtype=np.float32,
       config=NetworkConfig(buffer_size=10, seed=42),
       channel_factory=GEChannel,
   )

   env = gym.make("CartPole-v1")
   obs, _ = env.reset()
   central.reset()

   for step in range(200):
       central.receive_from("agent_0", obs, step)
       central.receive_from("agent_1", obs, step, packet_size=256)
       arrived = central.flush_and_update(step)
       obs, _, term, trunc, _ = env.step(env.action_space.sample())
       if term or trunc:
           obs, _ = env.reset()

   buf_0, mask_0 = central.get_buffer("agent_0")
   # buf_0.shape == (10, 4),  mask_0.shape == (10,)

Per-node observation shapes
----------------------------

Pass a list to give each node its own observation shape:

.. code-block:: python

   central = CentralNode(
       node_ids=["lidar", "camera"],
       obs_shape=[(2,), (8,)],    # list → per-node shapes
       obs_dtype=[np.float32, np.float32],
       config=NetworkConfig(buffer_size=8),
       channel_factory=GEChannel,
   )
