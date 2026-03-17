ObservationBuffer
=================

.. autoclass:: netrl.ObservationBuffer
   :members:
   :special-members: __init__
   :show-inheritance:

Semantics
---------

The buffer is a fixed-size circular window.  After ``maxlen`` consecutive
``add()`` calls, the oldest entry is silently overwritten.

.. code-block:: python

   from netrl import ObservationBuffer
   import numpy as np

   buf = ObservationBuffer(maxlen=4, shape=(3,), dtype=np.float32)

   buf.add(np.array([1., 2., 3.]))
   buf.add(None)                     # packet loss → zero-padded slot
   buf.add(np.array([4., 5., 6.]))

   obs, mask = buf.get_padded()
   # obs.shape  == (4, 3)
   # mask       == [False, True, False, True]  (oldest → newest)
   # obs[-1]    == [4., 5., 6.]   ← most recent real observation
   # obs[-2]    == [0., 0., 0.]   ← lost packet (zero fill)
   # obs[-3]    == [1., 2., 3.]
   # obs[-4]    == [0., 0., 0.]   ← unwritten slot (buffer not yet full)
