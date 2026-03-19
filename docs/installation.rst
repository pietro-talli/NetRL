Installation
============

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

* **Python** ≥ 3.10
* **GCC** ≥ 10 or **Clang** ≥ 11 (C++20, for both pybind11 extensions)
* **gymnasium** ≥ 0.29
* **numpy** ≥ 1.24

The ns-3 channel backends are optional.  Install only what you need.

Core package
------------

Install from source:

.. code-block:: bash

   git clone https://github.com/pietro-talli/NetRL.git
   cd NetRL
   pip install -e .

This does three things automatically:

1. Installs all Python dependencies (including ``ns3 ≥ 3.44``).
2. Compiles ``netcomm`` — the Gilbert–Elliott C++ pybind11 extension.
3. Detects the pip-installed ``ns3`` library and compiles ``netrl_ext`` — the
   **fast WiFi pybind11 extension** (:class:`~netrl.NS3WiFiChannelFast`).

After this single command, both the default :class:`~netrl.GEChannel` backend
and the fast :class:`~netrl.NS3WiFiChannelFast` backend are immediately
available — no extra build step required.

.. note::

   If ``ns3`` is not found or cannot be detected at build time, ``netrl_ext``
   is silently skipped and only ``netcomm`` is built.  Install ns3 later and
   re-run ``pip install -e .`` or ``python setup.py build_ext --inplace``.

Verify installation:

.. code-block:: python

   import netrl
   print(netrl.__version__)   # 0.2.0

ns-3 802.11a WiFi (fast — pybind11)
------------------------------------

.. note::

   **No extra build step needed.**  The fast WiFi extension is compiled
   automatically during ``pip install -e .`` (see above).

The fast backend runs the same 802.11a OFDM / CSMA/CA simulation as the
subprocess version, but as a Python C++ extension linked directly into the
interpreter process.  This eliminates subprocess-spawn and pipe-IPC overhead,
giving 15–20× better throughput.

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig, NS3WiFiChannelFastConfig

   env = NetworkedEnv(
       base_env,
       NetworkConfig(buffer_size=10, seed=42),
       channel_config=NS3WiFiChannelFastConfig(
           distance_m=20.0,
           step_duration_ms=2.0,
       ),
   )

Verify the extension was built:

.. code-block:: python

   import netrl_ext
   print(netrl_ext.NS3WiFiChannel.__doc__)

ns-3 802.11a WiFi backend (subprocess)
---------------------------------------

The subprocess ns-3 backend runs the simulation in a **separate process** and
communicates over stdin/stdout pipes.  It is slower than the pybind11 fast
backend but does not require a C++20 compiler beyond what ``pip install ns3``
provides.  Compile the binary once before use:

.. code-block:: bash

   bash src/build_ns3_sim.sh

Verify:

.. code-block:: bash

   printf 'QUIT\n' | timeout 30 src/ns3_wifi_sim 2>/dev/null | grep READY
   # Expected: READY

ns-3 5G mmWave backend
-----------------------

Requires the **ns3-mmwave** fork (ns-3 3.42 + mmWave module).

.. code-block:: bash

   bash src/build_ns3_mmwave_sim.sh

ns-3 5G-LENA NR backend
------------------------

Requires ns-3 with the **5G-LENA** contrib module installed.

.. code-block:: bash

   bash src/build_ns3_lena_sim.sh

Multi-UE WiFi backend
---------------------

Uses the same ns-3 installation as the single-UE WiFi backend.

.. code-block:: bash

   bash src/build_ns3_multi_ue_sim.sh

Verify:

.. code-block:: bash

   printf 'QUIT\n' | timeout 60 src/ns3_wifi_multi_ue_sim 2>/dev/null | grep READY
   # Expected: READY

Development extras
------------------

.. code-block:: bash

   pip install -e ".[dev]"   # pytest + matplotlib
