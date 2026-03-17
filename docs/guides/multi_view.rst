Multi-View Environments
========================

:class:`~netrl.MultiViewNetworkedEnv` extends the single-observer model to
**N independent sensors**, each with its own channel and observation buffer.
Sensors can have different observation shapes and may transmit on different
schedules or with different payload sizes.

.. contents:: On this page
   :local:
   :depth: 2

Motivation
----------

Many real systems are monitored by heterogeneous sensors:

* A camera (high-bandwidth, bursty)
* A lidar (medium bandwidth, periodic)
* An IMU (low-bandwidth, frequent)

Each sensor transmits over a shared or independent wireless link.  The central
node (controller) must fuse these streams, accounting for losses and delays on
each path.

The ``MultiViewModel`` abstraction
------------------------------------

:class:`~netrl.MultiViewModel` separates *what* each sensor observes from *how*
the transmission network is simulated.  You subclass it to define the sensor
observation function:

.. code-block:: python

   import numpy as np
   from netrl import MultiViewModel

   class MySensors(MultiViewModel):
       def observe(self, env, state):
           """
           Parameters
           ----------
           env   : gymnasium.Env   The wrapped environment (access render, etc.)
           state : np.ndarray      The raw observation returned by env.step().

           Returns
           -------
           Dict[observer_id → np.ndarray]
               One observation array per observer.  Each array must match the
               shape declared in obs_shapes.
           """
           return {
               "lidar":   state[:2].astype(np.float32),
               "camera":  np.random.randn(8).astype(np.float32),
               "imu":     state[2:].astype(np.float32),
           }

   model = MySensors(
       observer_ids=["lidar", "camera", "imu"],
       obs_shapes  =[(2,), (8,), (2,)],
       obs_dtypes  =[np.float32, np.float32, np.float32],
   )

.. note::

   ``observe()`` is called *every step* before any transmission decisions are
   made.  The observations it returns are used only for the observers that are
   active (non-masked) that step.

Constructing the environment
-----------------------------

.. code-block:: python

   import gymnasium as gym
   from netrl import NetworkConfig, MultiViewNetworkedEnv
   from netrl.channels.comm_channel import GEChannel

   env = MultiViewNetworkedEnv(
       gym.make("CartPole-v1"),
       config=NetworkConfig(buffer_size=8, loss_bad=0.25, seed=42),
       observer_ids=["lidar", "camera", "imu"],
       multi_view_model=model,
       channel_factory=GEChannel,   # one independent GE channel per observer
   )

Observation space
-----------------

The returned observation space is a nested ``Dict``:

.. code-block:: text

   gymnasium.spaces.Dict({
       "lidar": Dict({
           "observations": Box(shape=(8, 2), dtype=float32),
           "recv_mask":    MultiBinary(8),
       }),
       "camera": Dict({
           "observations": Box(shape=(8, 8), dtype=float32),
           "recv_mask":    MultiBinary(8),
       }),
       "imu": Dict({
           "observations": Box(shape=(8, 2), dtype=float32),
           "recv_mask":    MultiBinary(8),
       }),
   })

Using ``step()``
----------------

The ``step()`` method accepts two keyword-only arguments that give per-step
control over transmissions:

``transmit_mask``
   A ``Dict[str, bool]`` controlling which observers are active this step.
   ``None`` (default) means **all observers transmit**.  Absent keys default
   to ``True`` (opt-out semantics):

   .. code-block:: python

      # Only lidar transmits; camera and imu are silenced
      obs, r, term, trunc, info = env.step(
          action,
          transmit_mask={"lidar": True, "camera": False, "imu": False},
      )

``packet_sizes``
   A ``Dict[str, int]`` overriding the payload bytes per active observer.
   ``None`` (default) uses each channel's configured default:

   .. code-block:: python

      # lidar sends 256 bytes; camera sends 4096 bytes; imu uses default
      obs, r, term, trunc, info = env.step(
          action,
          packet_sizes={"lidar": 256, "camera": 4096},
      )

Both arguments can be combined:

.. code-block:: python

   obs, r, term, trunc, info = env.step(
       action,
       transmit_mask={"lidar": True, "camera": True, "imu": False},
       packet_sizes={"lidar": 128, "camera": 2048},
   )

.. important::

   :meth:`~netrl.CentralNode.flush_and_update` is called for **all**
   observers every step, regardless of ``transmit_mask``.  This guarantees
   that every buffer advances by exactly one slot per step; delayed packets
   from previous steps are still collected for observers that were silent
   this step.

The extended ``info`` dict
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Value
   * - ``"channel_info"``
     - ``Dict[observer_id → channel_info_dict]``
   * - ``"arrived_this_step"``
     - ``Dict[observer_id → bool]`` — packet arrived at central node
   * - ``"transmitted_this_step"``
     - ``Dict[observer_id → bool]`` — transmission was attempted

Using a shared 802.11a WiFi channel
-------------------------------------

Replace independent GE channels with a single ns-3 infrastructure BSS where
all observers compete for the wireless medium via CSMA/CA.  This requires
building the multi-UE binary first:

.. code-block:: bash

   bash src/build_ns3_multi_ue_sim.sh

.. code-block:: python

   from netrl import NS3WifiMultiUEConfig, make_multi_ue_wifi_factory

   factory = make_multi_ue_wifi_factory(
       NS3WifiMultiUEConfig(
           n_ues=3,                          # must equal len(observer_ids)
           distances_m=[10.0, 30.0, 60.0],  # distance per observer
           step_duration_ms=2.0,
           packet_size_bytes=128,
       )
   )

   env = MultiViewNetworkedEnv(
       gym.make("CartPole-v1"),
       NetworkConfig(buffer_size=8),
       observer_ids=["lidar", "camera", "imu"],
       multi_view_model=model,
       channel_factory=factory,
   )

When all three observers transmit in the same step they contend for the
channel via MAC-layer CSMA/CA backoff — the same way real Wi-Fi devices do.

Implementation tips
--------------------

Duty-cycling sensors
   Alternate which observers transmit to share channel capacity:

   .. code-block:: python

      for step in range(N):
          mask = {oid: (step % len(observer_ids) == i)
                  for i, oid in enumerate(observer_ids)}
          obs, r, term, trunc, info = env.step(action, transmit_mask=mask)

Adaptive packet sizes
   Transmit full observations when the channel is in the Good state, and
   reduced observations when in the Bad state:

   .. code-block:: python

      sizes = {
          oid: 256 if info["channel_info"][oid].get("state") == "GOOD" else 64
          for oid in observer_ids
      }
      obs, r, term, trunc, info = env.step(action, packet_sizes=sizes)

Accessing buffers directly
   Use :attr:`~netrl.MultiViewNetworkedEnv.central_node` to access raw
   buffers outside of the step loop:

   .. code-block:: python

      for oid in env.observer_ids:
          buf, mask = env.central_node.get_buffer(oid)
          print(f"{oid}: {mask.sum()} packets received in last {len(mask)} steps")
