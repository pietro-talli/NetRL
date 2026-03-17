Troubleshooting
================

.. contents:: On this page
   :local:
   :depth: 2

Build errors
------------

``ImportError: No module named 'netcomm'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C++ pybind11 extension was not built.  Run:

.. code-block:: bash

   pip install -e .

If the build fails, ensure you have GCC ≥ 10 or Clang ≥ 11 and pybind11:

.. code-block:: bash

   pip install pybind11
   g++ --version   # must be ≥ 10

``FileNotFoundError: ns3_wifi_sim binary not found``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ns-3 binary was not built.  Run the appropriate build script:

.. code-block:: bash

   bash src/build_ns3_sim.sh              # 802.11a WiFi
   bash src/build_ns3_mmwave_sim.sh       # 5G mmWave
   bash src/build_ns3_lena_sim.sh         # 5G-LENA NR
   bash src/build_ns3_multi_ue_sim.sh     # Multi-UE WiFi

If ns-3 is not found, install it:

.. code-block:: bash

   pip install ns3   # installs ns-3 ≥ 3.43

``ns3 installation not found`` (build script error)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The build scripts look for ns-3 in two locations:

1. ``pip install ns3`` — auto-detected via Python's ``importlib``.
2. A source build of ns3-mmwave at ``/home/dianalab/Projects/ns3-mmwave/``.

If you have ns-3 installed elsewhere, edit the ``NS3_MMWAVE_BUILD`` variable
in the relevant build script.

Runtime errors
--------------

``RuntimeError: ns3_wifi_sim did not emit READY within 30 s``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subprocess failed to initialise.  Possible causes:

* **Wrong binary**: verify the binary path matches the configured
  ``sim_binary`` option (default: ``src/ns3_wifi_sim``).
* **ns-3 shared libraries not found**: check that ``LD_LIBRARY_PATH``
  includes the ns-3 library directory, or that the binary was linked with
  ``-Wl,-rpath``.  The build scripts set ``-Wl,-rpath`` automatically.
* **Smoke test**: run the binary directly to see error output:

  .. code-block:: bash

     printf 'QUIT\n' | src/ns3_wifi_sim 2>&1 | head -20

``RuntimeError: subprocess stdin pipe broken``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ns-3 subprocess exited unexpectedly.  Inspect stderr:

.. code-block:: python

   # Print the last 20 stderr lines from the backend
   print(env.central_node.get_channel_info("agent_0"))
   # For multi-UE:
   channel = env.central_node._channels["ue_0"]
   print(channel._backend._drain_stderr())

Enable ns-3 logging by setting the ``NS_LOG`` environment variable before
starting Python:

.. code-block:: bash

   export NS_LOG="UdpSocket=level_all|prefix_time"
   python your_script.py

``TimeoutError: no response within N s (FLUSH)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ns-3 simulation is slower than the Python main loop.  Increase
``step_duration_ms`` so ns-3 has more virtual time per step, or reduce
the episode framerate.  A value of 2–20 ms is typical; 1 ms steps can
overload the scheduler for complex topologies.

Performance
-----------

``GEChannel`` throughput is much higher than ns-3 backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is expected.  The C++ GE channel operates at ~10 k–100 k steps/s
in Python.  ns-3 backends run in a separate process and are limited by
IPC round-trips; typical throughput is 50–500 steps/s depending on
``step_duration_ms`` and topology complexity.

To maximise ns-3 throughput:

* Use ``AsyncVectorEnv`` to run multiple environments in parallel on
  separate CPU cores.
* Increase ``step_duration_ms`` slightly — a bigger time window means
  more ns-3 work per IPC round-trip.
* Disable ns-3 PCAP tracing if you enabled it during debugging.

Observations
------------

``obs["recv_mask"]`` is all ``False`` for many steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check:

1. **delay_steps** (GE channel): if ``delay_steps=5``, observations
   arrive 5 steps late.  The first 5 steps will always have empty buffers.
2. **High loss**: increase ``tx_power_dbm``, decrease ``distance_m``, or
   lower ``loss_exponent``.
3. **step_duration_ms too small**: if the step window is shorter than
   the MAC retransmission time, packets time out at the ns-3 MAC layer.
   Try ``step_duration_ms=5.0`` or higher.

``obs["observations"]`` contains stale data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the intended behaviour.  When a slot has ``recv_mask[i] == False``,
the corresponding ``observations[i]`` is zero-padded (not a copy of the
previous observation).  Your agent should use ``recv_mask`` to identify
valid slots.

Multi-UE issues
---------------

``ValueError: Factory called for UE index N but n_ues=M``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of ``node_ids`` passed to :class:`~netrl.CentralNode` (or
``observer_ids`` in :class:`~netrl.MultiViewNetworkedEnv`) does not match
``NS3WifiMultiUEConfig.n_ues``.  They must be equal:

.. code-block:: python

   ns3_cfg = NS3WifiMultiUEConfig(n_ues=3, ...)
   factory  = make_multi_ue_wifi_factory(ns3_cfg)

   # ✓ correct: 3 observer_ids for n_ues=3
   env = MultiViewNetworkedEnv(..., observer_ids=["a","b","c"], channel_factory=factory)

   # ✗ wrong: 2 ids but n_ues=3
   env = MultiViewNetworkedEnv(..., observer_ids=["a","b"], channel_factory=factory)

Some UEs have much lower packet delivery than others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is physically realistic — UEs further from the AP experience more
path loss and more retransmissions.  Verify the ``distances_m`` list:

.. code-block:: python

   for oid in env.observer_ids:
       info = env.central_node.get_channel_info(oid)
       print(oid, info["distance_m"], "m")

Reduce distances or increase ``tx_power_dbm`` / decrease ``loss_exponent``
for the distant UEs.
