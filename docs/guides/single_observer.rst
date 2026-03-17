Single-Observer Environments
=============================

:class:`~netrl.NetworkedEnv` wraps any Gymnasium environment whose observation
space is a ``Box``.  A single observation path — one channel, one buffer — is
simulated between the environment and the agent.

.. contents:: On this page
   :local:
   :depth: 2

Observation space
-----------------

The original ``Box(obs_shape)`` is replaced by a ``Dict``:

.. code-block:: text

   gymnasium.spaces.Dict({
       "observations": Box(shape=(buffer_size, *obs_shape), dtype=obs_dtype),
       "recv_mask":    MultiBinary(buffer_size),
   })

The agent sees the last ``buffer_size`` delivery slots.  Slot ``[-1]`` is the
most recent; slot ``[0]`` is the oldest.  Slots where no packet arrived
(loss or delay) are zero-padded; ``recv_mask[i] == True`` means a real
observation occupies slot ``i``.

Per-step sequence
-----------------

For each call to ``env.step(action)``:

1. The wrapped environment is stepped: ``raw_obs, reward, term, trunc, info``.
2. ``raw_obs`` is serialised into a UDP-payload packet and handed to the channel.
3. The channel is flushed: packets due at this step are delivered (or dropped).
4. The observation buffer is updated with the arrived packet (or ``None``).
5. The Dict observation (buffer + mask) is returned.

.. note::

   The channel simulates the *forward* path only (sensor → central node).
   The return path (controller → actuator) is instantaneous.

Construction
------------

.. code-block:: python

   from netrl import NetworkedEnv, NetworkConfig

   env = NetworkedEnv(
       base_env,
       config=NetworkConfig(
           p_gb=0.10,       # Good → Bad
           p_bg=0.30,       # Bad  → Good
           loss_good=0.01,
           loss_bad=0.20,
           delay_steps=2,
           buffer_size=10,
           seed=42,
       ),
       channel_config=None,   # None → Gilbert–Elliott (default)
   )

Selecting a different channel backend is done via ``channel_config``:

.. code-block:: python

   from netrl import NS3WifiConfig
   env = NetworkedEnv(base_env, config,
                      channel_config=NS3WifiConfig(distance_m=30.0))

See :doc:`channels` for details on all available backends.

Using ``step()``
----------------

.. code-block:: python

   obs, reward, term, trunc, info = env.step(action)

   # Optional: override the packet payload size for this step only
   obs, reward, term, trunc, info = env.step(action, packet_size=256)

The ``info`` dictionary is augmented with:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``"channel_info"``
     - ``dict`` from :meth:`CommChannel.get_channel_info` — includes
       ``"state"`` (``"GOOD"`` / ``"BAD"`` for GE), ``"pending_count"``, etc.
   * - ``"arrived_this_step"``
     - ``bool`` — ``True`` if a packet arrived at the central node this step.

Resetting
---------

.. code-block:: python

   obs, info = env.reset()

This resets the wrapped environment **and** calls ``central_node.reset()``,
which clears the channel queues, resets the GE Markov state, and zeroes the
observation buffer.  For ns-3 backends the subprocess is fully reinitialised.

Training with Stable-Baselines3
---------------------------------

:class:`~netrl.NetworkedEnv` is a standard ``gymnasium.Wrapper`` and works with
any SB3 policy that accepts ``MultiInputPolicy``:

.. code-block:: python

   from stable_baselines3 import PPO
   from netrl import NetworkedEnv, NetworkConfig

   env = NetworkedEnv(gym.make("CartPole-v1"), NetworkConfig(buffer_size=10))
   model = PPO("MultiInputPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)

Parallel environments
----------------------

Use ``gymnasium.vector.AsyncVectorEnv`` to run multiple independent
environments (each with its own channel subprocess) in parallel:

.. code-block:: python

   from gymnasium.vector import AsyncVectorEnv
   from netrl import NetworkedEnv, NetworkConfig, NS3WifiConfig

   def make_env(seed):
       def _fn():
           return NetworkedEnv(
               gym.make("CartPole-v1"),
               NetworkConfig(buffer_size=10, seed=seed),
               channel_config=NS3WifiConfig(distance_m=30.0, step_duration_ms=2.0),
           )
       return _fn

   vec_env = AsyncVectorEnv([make_env(i) for i in range(4)])
   obs, info = vec_env.reset()
   # obs["observations"].shape == (4, 10, 4)
