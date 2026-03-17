Installation
============

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

* **Python** ≥ 3.10
* **GCC** ≥ 10 or **Clang** ≥ 11 (for the C++ pybind11 extension)
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

This builds the ``netcomm`` C++17 pybind11 extension (Gilbert–Elliott channel
core) automatically via ``setup.py``.  No external simulator is needed for the
default :class:`~netrl.GEChannel` backend.

Verify installation:

.. code-block:: python

   import netrl
   print(netrl.__version__)   # 0.2.0

ns-3 802.11a WiFi backend
-------------------------

Requires `ns-3 <https://www.nsnam.org>`_ ≥ 3.43 (pip-installable) **or** an
ns3-mmwave source build.

.. tab-set::

   .. tab-item:: pip install (recommended)

      .. code-block:: bash

         pip install ns3          # installs ns-3.44 or later
         bash src/build_ns3_sim.sh

   .. tab-item:: ns3-mmwave source

      .. code-block:: bash

         # Build ns3-mmwave first (see ns3-mmwave docs)
         # Then:
         bash src/build_ns3_sim.sh

Verify the binary:

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
