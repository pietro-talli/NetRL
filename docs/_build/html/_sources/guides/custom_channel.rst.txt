Writing a Custom Channel
=========================

All channel backends in NetRL implement the :class:`~netrl.CommChannel`
abstract base class.  Subclassing it lets you plug in any channel model —
physical-layer simulators, trace-driven replay, hardware-in-the-loop — without
modifying :class:`~netrl.CentralNode` or either environment wrapper.

.. contents:: On this page
   :local:
   :depth: 2

The CommChannel interface
--------------------------

.. code-block:: python

   from netrl.channels.comm_channel import CommChannel
   import numpy as np
   from typing import List, Optional, Tuple

   class CommChannel:
       def transmit(
           self,
           obs: np.ndarray,
           step: int,
           packet_size: Optional[int] = None,
       ) -> None: ...

       def flush(self, step: int) -> List[Tuple[int, np.ndarray]]: ...

       def reset(self) -> None: ...

       def get_channel_info(self) -> dict: ...

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Contract
   * - ``transmit``
     - Called once per step when an observer transmits.  Schedule the packet
       for future delivery (or drop it).  Must never raise on valid inputs.
   * - ``flush``
     - Called once per step for *every* observer.  Return a list of
       ``(arrival_step, obs)`` tuples for packets due at or before ``step``.
       Return an empty list if none are ready.
   * - ``reset``
     - Clear all internal state.  Called on ``env.reset()``.
   * - ``get_channel_info``
     - Return a diagnostic ``dict``.  Must include ``"state"`` and
       ``"pending_count"`` keys; add any extra keys you like.

Minimal example: Bernoulli channel
------------------------------------

A stateless channel that drops each packet independently with fixed
probability and adds a fixed delay:

.. code-block:: python

   import numpy as np
   from typing import List, Optional, Tuple
   from netrl.channels.comm_channel import CommChannel
   from netrl.channels.network_config import NetworkConfig

   class BernoulliChannel(CommChannel):
       """Independent packet loss with fixed probability and fixed delay."""

       def __init__(self, config: NetworkConfig, loss_prob: float = 0.1) -> None:
           self._loss  = loss_prob
           self._delay = config.delay_steps
           self._rng   = np.random.default_rng(config.seed)
           self._queue: List[Tuple[int, np.ndarray]] = []

       def transmit(
           self,
           obs: np.ndarray,
           step: int,
           packet_size: Optional[int] = None,
       ) -> None:
           if self._rng.random() >= self._loss:
               self._queue.append((step + self._delay, obs.copy()))

       def flush(self, step: int) -> List[Tuple[int, np.ndarray]]:
           due, remaining = [], []
           for arrival, obs in self._queue:
               (due if arrival <= step else remaining).append((arrival, obs))
           self._queue = remaining
           return due

       def reset(self) -> None:
           self._queue.clear()

       def get_channel_info(self) -> dict:
           return {
               "state":         "BERNOULLI",
               "pending_count": len(self._queue),
               "loss_prob":     self._loss,
           }

Using the custom channel
--------------------------

Because the factory signature is ``Callable[[NetworkConfig], CommChannel]``,
you can close over any extra parameters:

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig

   factory = lambda nc: BernoulliChannel(nc, loss_prob=0.15)

   env = NetworkedEnv(
       gym.make("CartPole-v1"),
       NetworkConfig(delay_steps=3, buffer_size=10),
       channel_factory=factory,
   )

For :class:`~netrl.MultiViewNetworkedEnv` or :class:`~netrl.CentralNode`
pass it directly:

.. code-block:: python

   central = CentralNode(
       node_ids=["a", "b"],
       obs_shape=(4,),
       obs_dtype=np.float32,
       config=NetworkConfig(buffer_size=10),
       channel_factory=factory,
   )

Subprocess-backed channel
--------------------------

For channels that delegate to an external process (as all ns-3 backends do),
the same interface applies.  The key design points from the existing backends:

1. **Start the subprocess in** ``__init__``.  Read until ``READY``.
2. **``transmit``** sends a command and consumes the ``OK`` response.
3. **``flush``** sends ``FLUSH <step>``, consumes the response, and returns
   matched observations from an internal ``_pending`` dict.
4. **``reset``** sends ``RESET``, waits for ``OK``, and clears ``_pending``.

See ``netrl/channels/ns3_channel.py`` for a complete production example.

Factory pattern for shared subprocesses
-----------------------------------------

When multiple channel instances must share one subprocess (as with
:func:`~netrl.make_multi_ue_wifi_factory`), use a factory closure that
creates the subprocess once and returns lightweight proxy objects:

.. code-block:: python

   class _Backend:
       """Owns the subprocess."""
       ...

   class _ProxyChannel(CommChannel):
       """Lightweight proxy for one 'lane' inside _Backend."""
       def __init__(self, uid: int, backend: _Backend): ...

   def make_factory(cfg):
       backend  = _Backend(cfg)
       counter  = [0]
       def factory(net_cfg):
           uid = counter[0]; counter[0] += 1
           return _ProxyChannel(uid, backend)
       return factory
