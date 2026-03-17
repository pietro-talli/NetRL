MultiViewModel
==============

:class:`~netrl.MultiViewModel` is an abstract base class that decouples the
*observation function* (what each sensor measures) from the network simulation.
Users subclass it to define per-sensor observations.

.. autoclass:: netrl.MultiViewModel
   :members:
   :special-members: __init__
   :show-inheritance:

Usage pattern
-------------

.. code-block:: python

   import numpy as np
   from netrl import MultiViewModel

   class MySensors(MultiViewModel):
       def observe(self, env, state):
           """Return one observation array per observer_id."""
           return {
               self.observer_ids[0]: state[:2].astype(np.float32),
               self.observer_ids[1]: np.random.randn(8).astype(np.float32),
           }

   model = MySensors(
       observer_ids=["lidar", "camera"],
       obs_shapes  =[(2,), (8,)],
       obs_dtypes  =[np.float32, np.float32],
   )

The ``model.spaces`` dictionary holds the per-observer ``Box`` spaces
(single-step shapes, without the buffer dimension):

.. code-block:: python

   model.spaces["lidar"]   # Box(shape=(2,), dtype=float32)
   model.spaces["camera"]  # Box(shape=(8,), dtype=float32)

.. note::

   :class:`~netrl.MultiViewNetworkedEnv` wraps each observer's space with
   the buffer dimension to produce ``Box(shape=(buffer_size, *obs_shape))``.
