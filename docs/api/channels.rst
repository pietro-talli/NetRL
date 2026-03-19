Channels
=========

.. contents:: On this page
   :local:
   :depth: 1

CommChannel (abstract base)
----------------------------

.. autoclass:: netrl.CommChannel
   :members:
   :show-inheritance:

----

NetworkConfig
-------------

.. autoclass:: netrl.NetworkConfig
   :members:
   :show-inheritance:

.. list-table:: Parameter reference
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Description
   * - ``p_gb``
     - 0.1
     - Good → Bad transition probability per step.
   * - ``p_bg``
     - 0.3
     - Bad → Good transition probability per step.
   * - ``loss_good``
     - 0.01
     - Packet loss probability in the Good state.
   * - ``loss_bad``
     - 0.20
     - Packet loss probability in the Bad state.
   * - ``delay_steps``
     - 3
     - Fixed one-way propagation delay (integer steps).
   * - ``buffer_size``
     - 10
     - Number of slots in the observation sliding window.
   * - ``seed``
     - 42
     - RNG seed for the C++ GE channel core.

----

GEChannel
---------

.. autoclass:: netrl.GEChannel
   :members:
   :show-inheritance:

PerfectChannel
--------------

.. autoclass:: netrl.PerfectChannel
   :members:
   :show-inheritance:

----

NS3WiFiChannelFastConfig
------------------------

.. autoclass:: netrl.NS3WiFiChannelFastConfig
   :members:
   :show-inheritance:

NS3WiFiChannelFast
------------------

.. autoclass:: netrl.NS3WiFiChannelFast
   :members:
   :show-inheritance:

----

NS3WifiConfig
-------------

.. autoclass:: netrl.NS3WifiConfig
   :members:
   :show-inheritance:

NS3WifiChannel
--------------

.. autoclass:: netrl.NS3WifiChannel
   :members:
   :show-inheritance:

----

NS3MmWaveConfig
---------------

.. autoclass:: netrl.NS3MmWaveConfig
   :members:
   :show-inheritance:

NS3MmWaveChannel
----------------

.. autoclass:: netrl.NS3MmWaveChannel
   :members:
   :show-inheritance:

----

NS3LenaConfig
-------------

.. autoclass:: netrl.NS3LenaConfig
   :members:
   :show-inheritance:

NS3LenaChannel
--------------

.. autoclass:: netrl.NS3LenaChannel
   :members:
   :show-inheritance:

----

NS3WifiMultiUEConfig
---------------------

.. autoclass:: netrl.NS3WifiMultiUEConfig
   :members:
   :show-inheritance:

make_multi_ue_wifi_factory
---------------------------

.. autofunction:: netrl.make_multi_ue_wifi_factory

NS3WifiUEChannel
-----------------

.. autoclass:: netrl.NS3WifiUEChannel
   :members:
   :show-inheritance:

NS3WifiMultiUEBackend
----------------------

.. autoclass:: netrl.NS3WifiMultiUEBackend
   :members:
   :show-inheritance:
