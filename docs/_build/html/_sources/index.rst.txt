NetRL Documentation
===================

**NetRL** is a Python library that wraps `Gymnasium <https://gymnasium.farama.org>`_
environments with realistic communication-channel models.  Instead of receiving
observations directly, an RL agent or central node receives them through a
simulated wireless link — complete with packet loss, propagation delay, retransmissions,
and medium contention.  The agent sees a sliding-window buffer of past observations
rather than the raw current state.

.. code-block:: python

   import gymnasium as gym
   from netrl import NetworkedEnv, NetworkConfig

   env = NetworkedEnv(
       gym.make("CartPole-v1"),
       NetworkConfig(loss_bad=0.3, delay_steps=2, buffer_size=10),
   )
   obs, info = env.reset()
   # obs["observations"].shape == (10, 4)   ← sliding window
   # obs["recv_mask"].shape    == (10,)     ← delivery flags

.. grid:: 2

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: quickstart
      :link-type: doc

      Get running in five minutes with a single-observer networked environment.

   .. grid-item-card:: :octicon:`book` Guides
      :link: guides/index
      :link-type: doc

      Step-by-step guides for single-observer, multi-view, channel selection,
      and writing custom channel backends.

.. grid:: 2

   .. grid-item-card:: :octicon:`code-square` API Reference
      :link: api/index
      :link-type: doc

      Complete class and function reference auto-generated from source.

   .. grid-item-card:: :octicon:`tools` Reference
      :link: reference/index
      :link-type: doc

      Architecture overview, ns-3 protocol spec, and troubleshooting.

Channel backends
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Backend
     - Class
     - Description
   * - Gilbert–Elliott (default)
     - :class:`~netrl.GEChannel`
     - Two-state Markov loss model, C++ core, no external dependencies.
   * - 802.11a WiFi
     - :class:`~netrl.NS3WifiChannel`
     - CSMA/CA MAC, log-distance path loss, via ns-3 subprocess.
   * - 5G mmWave
     - :class:`~netrl.NS3MmWaveChannel`
     - 28 GHz mmWave EPC (3GPP TR 38.901), via ns-3-mmwave subprocess.
   * - 5G NR (5G-LENA)
     - :class:`~netrl.NS3LenaChannel`
     - NR with configurable numerology and beamforming, via 5G-LENA.
   * - Multi-UE WiFi
     - :class:`~netrl.NS3WifiUEChannel`
     - N UEs sharing one 802.11a infrastructure BSS; realistic contention.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
