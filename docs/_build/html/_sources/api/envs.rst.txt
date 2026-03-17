Environment Wrappers
=====================

.. contents:: On this page
   :local:
   :depth: 1

NetworkedEnv
------------

.. autoclass:: netrl.NetworkedEnv
   :members:
   :special-members: __init__
   :show-inheritance:

.. rubric:: Observation space

The original ``Box(obs_shape)`` is replaced with:

.. code-block:: text

   Dict({
       "observations": Box(shape=(buffer_size, *obs_shape), dtype=obs_dtype),
       "recv_mask":    MultiBinary(buffer_size),
   })

.. rubric:: Extended ``info`` keys

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Description
   * - ``"channel_info"``
     - ``dict`` from :meth:`~netrl.CommChannel.get_channel_info`.
   * - ``"arrived_this_step"``
     - ``bool`` — packet arrived this step.

----

MultiViewNetworkedEnv
----------------------

.. autoclass:: netrl.MultiViewNetworkedEnv
   :members:
   :special-members: __init__
   :show-inheritance:

.. rubric:: Observation space

.. code-block:: text

   Dict({
       "<observer_id>": Dict({
           "observations": Box(shape=(buffer_size, *obs_shape_i), dtype=obs_dtype_i),
           "recv_mask":    MultiBinary(buffer_size),
       }),
       ...
   })

.. rubric:: Extended ``info`` keys

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Description
   * - ``"channel_info"``
     - ``Dict[observer_id → channel_info_dict]``
   * - ``"arrived_this_step"``
     - ``Dict[observer_id → bool]``
   * - ``"transmitted_this_step"``
     - ``Dict[observer_id → bool]``
