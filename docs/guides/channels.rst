Choosing a Channel Backend
===========================

NetRL ships with five channel backends.  All implement the same
:class:`~netrl.CommChannel` interface, so you can swap channels without
changing any other code.

.. contents:: On this page
   :local:
   :depth: 2

Backend overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 15 48

   * - Backend
     - Requires
     - Subprocess
     - Best for
   * - :class:`~netrl.GEChannel`
     - nothing
     - no
     - Fast baseline; prototyping; reproducible sweeps.
   * - :class:`~netrl.PerfectChannel`
     - nothing
     - no
     - Debugging (no loss, no delay).
   * - :class:`~netrl.NS3WiFiChannelFast` ⚡
     - ``pip install ns3``
     - no
     - 802.11a single-link; **15–20× faster** than subprocess; built automatically.
   * - :class:`~netrl.NS3WifiChannel`
     - ns-3 ≥ 3.43
     - yes
     - Realistic 802.11a single-link; variable distance, power, path-loss.
   * - :class:`~netrl.NS3MmWaveChannel`
     - ns3-mmwave
     - yes
     - 5G mmWave (28 GHz) EPC; LOS/NLOS 3GPP TR 38.901 path-loss.
   * - :class:`~netrl.NS3LenaChannel`
     - 5G-LENA ns-3
     - yes
     - 5G NR with configurable numerology and NR beamforming.
   * - Multi-UE WiFi
     - ns-3 ≥ 3.43
     - yes (shared)
     - Multiple sensors contending for one 802.11a BSS; realistic CSMA/CA.

Gilbert–Elliott channel
-----------------------

The default backend.  A two-state Markov chain alternates between a
*Good* state (low loss) and a *Bad* state (high loss).  Implemented in
C++ via pybind11 for speed.

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(
           p_gb=0.10,       # Good → Bad transition probability per step
           p_bg=0.30,       # Bad  → Good
           loss_good=0.01,  # packet loss rate in Good state
           loss_bad=0.20,   # packet loss rate in Bad state
           delay_steps=2,   # fixed propagation delay (integer steps)
           buffer_size=10,
           seed=42,
       ),
   )

All GE parameters, including delay and buffer size, come from
:class:`~netrl.NetworkConfig`.

Perfect channel
---------------

Lossless, zero-delay channel useful for verifying that your RL logic
is correct before adding network effects:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, PerfectChannel

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=1),
       channel_factory=PerfectChannel,   # no channel_config needed
   )

.. note::

   When using :class:`~netrl.NetworkedEnv`, pass ``channel_factory`` directly.
   When using :class:`~netrl.MultiViewNetworkedEnv` or
   :class:`~netrl.CentralNode`, pass the factory via the ``channel_factory``
   parameter.

ns-3 802.11a WiFi (fast — pybind11) ⚡
--------------------------------------

The recommended 802.11a backend.  Runs the same OFDM / CSMA/CA simulation as
the subprocess version but as a Python C++ extension linked directly into the
interpreter process — eliminating subprocess spawn and pipe-IPC overhead
entirely.

**No build step needed** — the extension is compiled automatically by
``pip install -e .`` when ``ns3`` is pip-installed.

**Usage**:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3WiFiChannelFastConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=10, seed=42),
       channel_config=NS3WiFiChannelFastConfig(
           distance_m=30.0,         # STA–AP distance (metres)
           step_duration_ms=2.0,    # ns-3 time window per env step
           tx_power_dbm=20.0,       # transmit power (dBm)
           loss_exponent=3.0,       # log-distance path-loss exponent
           max_retries=7,           # MAC retry limit
           packet_size_bytes=64,    # default UDP payload (bytes)
       ),
   )

.. list-table:: NS3WiFiChannelFastConfig reference
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``distance_m``
     - 10.0
     - Euclidean distance from STA to AP in metres.
   * - ``step_duration_ms``
     - 1.0
     - ns-3 simulation time allocated to each env step (ms).
   * - ``tx_power_dbm``
     - 20.0
     - STA transmit power in dBm.
   * - ``loss_exponent``
     - 3.0
     - Path-loss exponent: 2 = free-space, 3 = mixed, 4 = dense indoor.
   * - ``max_retries``
     - 7
     - Maximum MAC frame retransmissions.
   * - ``packet_size_bytes``
     - 64
     - UDP payload size in bytes.

ns-3 802.11a WiFi (subprocess)
-------------------------------

A single-link 802.11a network between a station (STA) and an access point
(AP).  Packet loss is determined by real CSMA/CA MAC behaviour under the
configured path-loss model, not a stochastic approximation.

**Build** (once):

.. code-block:: bash

   bash src/build_ns3_sim.sh

**Usage**:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3WifiConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=10),
       channel_config=NS3WifiConfig(
           distance_m=30.0,         # STA–AP distance (metres)
           step_duration_ms=2.0,    # ns-3 time window per env step
           tx_power_dbm=20.0,       # transmit power (dBm)
           loss_exponent=3.0,       # log-distance path-loss exponent
           max_retries=7,           # MAC retry limit
           packet_size_bytes=64,    # default UDP payload (bytes)
       ),
   )

.. list-table:: NS3WifiConfig reference
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``distance_m``
     - 10.0
     - Euclidean distance from STA to AP in metres.
   * - ``step_duration_ms``
     - 1.0
     - ns-3 simulation time allocated to each env step (ms).
       Increase for environments with coarse time steps.
   * - ``tx_power_dbm``
     - 20.0
     - STA transmit power in dBm.
   * - ``loss_exponent``
     - 3.0
     - Path-loss exponent: 2 = free-space, 3 = mixed, 4 = dense indoor.
   * - ``max_retries``
     - 7
     - Maximum MAC frame retransmissions.
   * - ``packet_size_bytes``
     - 64
     - Default UDP payload; overridden per-step via ``packet_size``.
   * - ``max_pending_steps``
     - 200
     - Steps after which an unacknowledged packet is expired.

ns-3 5G mmWave
--------------

Uses the ns3-mmwave fork to simulate a 5G EPC with mmWave PHY including
LOS/NLOS switching and 3GPP TR 38.901 path-loss models.

**Build** (once):

.. code-block:: bash

   bash src/build_ns3_mmwave_sim.sh

**Usage**:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3MmWaveConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=10),
       channel_config=NS3MmWaveConfig(
           distance_m=200.0,
           frequency_ghz=28.0,
           bandwidth_ghz=0.5,
           step_duration_ms=20.0,
           harq_enabled=True,
           rlc_am_enabled=True,
           scenario="UMa",
       ),
   )

.. note::

   The mmWave backend requires a **500 ms warm-up** for UE attachment and
   bearer establishment.  The subprocess prints ``READY`` only after this
   warm-up; the Python side waits up to 60 s.

ns-3 5G-LENA NR
----------------

Uses the 5G-LENA contrib module for ns-3.  Supports configurable NR
numerology (subcarrier spacing) and full NR beamforming.

**Build** (once):

.. code-block:: bash

   bash src/build_ns3_lena_sim.sh

**Usage**:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3LenaConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=10),
       channel_config=NS3LenaConfig(
           distance_m=400.0,
           frequency_ghz=6.0,
           bandwidth_ghz=0.2,
           numerology=3,         # subcarrier spacing: 120 kHz
           scenario="UMa",
           step_duration_ms=20.0,
           shadowing_enabled=True,
       ),
   )

Multi-UE 802.11a WiFi
----------------------

A single ns-3 infrastructure BSS with N STAs sharing one AP.  All UEs
contend for the same channel via CSMA/CA, modelling medium congestion.
This backend is the natural choice for :class:`~netrl.MultiViewNetworkedEnv`.

**Build** (once):

.. code-block:: bash

   bash src/build_ns3_multi_ue_sim.sh

**Usage** with ``MultiViewNetworkedEnv``:

.. code-block:: python

   from netrl import (
       NetworkConfig, MultiViewNetworkedEnv,
       NS3WifiMultiUEConfig, make_multi_ue_wifi_factory,
   )

   factory = make_multi_ue_wifi_factory(
       NS3WifiMultiUEConfig(
           n_ues=3,
           distances_m=[10.0, 30.0, 60.0],
           step_duration_ms=2.0,
           tx_power_dbm=20.0,
           loss_exponent=3.0,
           max_retries=7,
           packet_size_bytes=64,
       )
   )

   env = MultiViewNetworkedEnv(
       base_env, config,
       observer_ids=["ue_0", "ue_1", "ue_2"],
       multi_view_model=my_model,
       channel_factory=factory,
   )

**Direct usage with** ``CentralNode``:

.. code-block:: python

   from netrl import CentralNode, NetworkConfig
   from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory
   import numpy as np

   factory = make_multi_ue_wifi_factory(
       NS3WifiMultiUEConfig(n_ues=2, distances_m=[10.0, 40.0])
   )
   central = CentralNode(
       node_ids=["ue_0", "ue_1"],
       obs_shape=(4,),
       obs_dtype=np.float32,
       config=NetworkConfig(buffer_size=10),
       channel_factory=factory,
   )

.. list-table:: NS3WifiMultiUEConfig reference
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``n_ues``
     - 2
     - Number of UE nodes (STAs).
   * - ``distances_m``
     - ``[10.0, 10.0]``
     - Distance per UE to the AP.  Shorter list is extended by repeating last.
   * - ``step_duration_ms``
     - 1.0
     - ns-3 time window per env step (ms).
   * - ``tx_power_dbm``
     - 20.0
     - Transmit power of every STA (dBm).
   * - ``loss_exponent``
     - 3.0
     - Log-distance path-loss exponent.
   * - ``max_retries``
     - 7
     - MAC retry limit per frame.
   * - ``packet_size_bytes``
     - 64
     - Default UDP payload (bytes); minimum 8.
   * - ``max_pending_steps``
     - 200
     - Steps before an unacknowledged packet is expired.
