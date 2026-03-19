Architecture & Concepts
========================

.. contents:: On this page
   :local:
   :depth: 2

System overview
---------------

NetRL intercepts the observation path between a Gymnasium environment and an
RL agent.  Instead of the agent receiving the state directly, the observation
is *transmitted* through a configurable channel model that introduces loss,
delay, and (for ns-3 backends) realistic wireless contention:

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │  Gymnasium step loop                                 │
   │                                                      │
   │  env.step(action)                                    │
   │    │                                                 │
   │    ▼                                                 │
   │  raw_obs ──► CommChannel.transmit()                 │
   │                     │  (loss / delay)               │
   │                     ▼                               │
   │  CommChannel.flush() ──► ObservationBuffer.add()   │
   │                               │                     │
   │                               ▼                     │
   │  agent ◄── Dict{"observations", "recv_mask"}        │
   └─────────────────────────────────────────────────────┘

For :class:`~netrl.MultiViewNetworkedEnv`, N independent paths run in
parallel — one per observer — managed by a single :class:`~netrl.CentralNode`:

.. code-block:: text

   raw_obs ──► MultiViewModel.observe() ──► {obs_0, obs_1, ..., obs_N}
                                                │
               ┌──────────────────────────────┬─┴──────────────────────┐
               │ Observer 0                   │ Observer N             │
               │ CommChannel.transmit(obs_0)  │ CommChannel.transmit() │
               │       ↓                      │       ↓                │
               │ CommChannel.flush()          │ CommChannel.flush()    │
               │       ↓                      │       ↓                │
               │ ObservationBuffer.add()      │ ObservationBuffer.add()│
               └──────────────────────────────┴────────────────────────┘
                               │
                               ▼
               Dict{"obs_0": {...}, "obs_1": {...}, ...}

Class hierarchy
---------------

.. code-block:: text

   CommChannel  (ABC)
   ├── GEChannel           — Markov chain; C++ core (netcomm extension)
   ├── PerfectChannel      — lossless; zero-delay
   ├── NS3WiFiChannelFast  — 802.11a ad-hoc; pybind11 in-process (netrl_ext)
   ├── NS3WifiChannel      — 802.11a ad-hoc; ns-3 subprocess
   ├── NS3MmWaveChannel    — 5G mmWave EPC; ns-3 subprocess
   ├── NS3LenaChannel      — 5G NR; ns-3 subprocess
   └── NS3WifiUEChannel    — per-UE proxy; shared NS3WifiMultiUEBackend

   ObservationBuffer       — fixed-size circular buffer + recv_mask

   CentralNode             — Dict[node_id → CommChannel + ObservationBuffer]

   gym.Wrapper
   ├── NetworkedEnv        — single observer; owns one CentralNode
   └── MultiViewNetworkedEnv — N observers; owns one CentralNode

Timing model
------------

Time is discretised into integer **env steps**.  Step ``t`` occupies ns-3
simulation time ``[t · step_ms, (t+1) · step_ms)``.

``transmit(obs, step=t)``
   The packet carrying observation ``obs`` is scheduled to be *sent* at
   ``t · step_ms + ε`` (a tiny offset into the step window).

``flush(step=t)``
   The ns-3 simulator is advanced to ``(t+1) · step_ms``.  Any packets whose
   receive callback fired during ``[t · step_ms, (t+1) · step_ms)`` are
   returned as the result.

For the Gilbert–Elliott backend there is no real-time simulation: ``transmit``
rolls a Markov state transition and samples a Bernoulli loss; ``flush`` pops
all packets whose ``arrival_step ≤ step`` from an in-memory deque.

Persisted simulation state
--------------------------

The ns-3 subprocess backends run *continuously* across steps.
``Simulator::Run()`` is called once per ``FLUSH`` with an increasing
stop-time.  The pybind11 fast backend (:class:`~netrl.NS3WiFiChannelFast`)
operates in the same way — the same NS3 simulator object lives inside the
Python process and is advanced in-place each step.  In both cases:

* MAC backoff counters, retry queues, and association state persist between steps.
* A ``RESET`` (triggered by ``env.reset()``) calls ``Simulator::Destroy()``
  and rebuilds the topology from scratch.

Warm-up period
~~~~~~~~~~~~~~

Infrastructure-mode backends (Multi-UE WiFi, mmWave, 5G-LENA) require a
warm-up phase before the first ``READY``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Backend
     - Warm-up
     - ``READY`` timeout
   * - 802.11a (ad-hoc, single STA)
     - 310 ms (3 beacon intervals)
     - 30 s
   * - 802.11a (infrastructure, N STAs)
     - 500 ms (association)
     - 60 s
   * - 5G mmWave / 5G-LENA
     - 500 ms (UE attach + bearer)
     - 60 s

Gilbert–Elliott channel model
-------------------------------

The GE channel is a two-state hidden Markov model:

.. code-block:: text

       p_gb               p_bg
   ┌─────────────────────────┐
   │                         │
   ▼                         │
   GOOD  ──────────────────► BAD
   loss_good                 loss_bad

At each ``transmit()`` call:

1. The Markov state is updated: transition with probability ``p_gb`` (Good→Bad)
   or ``p_bg`` (Bad→Good).
2. The packet is dropped with ``loss_good`` (Good state) or ``loss_bad``
   (Bad state).
3. If not dropped, the packet is queued with ``arrival_step = step + delay_steps``.

The C++ implementation uses a Mersenne Twister (``std::mt19937_64``) seeded
at construction.

ns-3 subprocess protocol
-------------------------

All ns-3 backends use an identical line-oriented stdin/stdout protocol:

**Python → subprocess**

.. code-block:: text

   TRANSMIT <step_id> <pkt_size>          # single-UE backends
   TRANSMIT <ue_id> <step_id> <pkt_size>  # multi-UE backend
   FLUSH    <step_id>
   RESET
   QUIT

**Subprocess → Python**

.. code-block:: text

   READY                          # once, at startup
   OK                             # ACK for TRANSMIT / RESET
   RECV <id1> <id2> ...           # single-UE: space-separated step_ids
   RECV <ue_id>:<step_id> ...     # multi-UE:  ue_id:step_id pairs
   ERROR <message>

The Python side stores the observation in a ``_pending`` dict keyed by
``step_id`` (or ``(ue_id, step_id)``).  On a successful ``FLUSH``, the
received ids are looked up to retrieve the original NumPy arrays.

Observation buffer semantics
------------------------------

Each :class:`~netrl.ObservationBuffer` is a fixed-size circular window.
``add(obs_or_None)`` advances by one slot *every step*, whether or not a
packet arrived:

.. code-block:: text

   step 0: transmit → arrives at step 2 (delay_steps=2)
   step 1: transmit → arrives at step 3
   step 2: flush → obs from step 0 arrives → buffer[-1] = obs_0
   step 3: flush → obs from step 1 arrives → buffer[-1] = obs_1
               buffer[-2] = obs_0

``get_padded()`` always returns ``(obs_array, recv_mask)`` of shape
``(maxlen, *obs_shape)`` and ``(maxlen,)``.  Unwritten or lost-packet slots
contain zero arrays with ``recv_mask == False``.

Strategy pattern
-----------------

:class:`~netrl.CentralNode` uses the **Strategy pattern** for channel
selection.  The ``channel_factory`` parameter is a
``Callable[[NetworkConfig], CommChannel]`` called once per node.  To add a
new backend:

1. Subclass :class:`~netrl.CommChannel` and implement the four methods.
2. Create a config dataclass with a ``validate()`` method.
3. Pass ``channel_factory=YourChannel`` to :class:`~netrl.CentralNode` or
   either environment wrapper.
