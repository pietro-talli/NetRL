import numpy as np


class ObservationBuffer:
    """
    Time-slot based circular buffer for storing RL observations.

    Each slot in the buffer represents a specific time step in the window
    [current_step - maxlen + 1, ..., current_step]. The recv mask indicates
    which time steps have received their observations.

    This allows delays to be visible: if delay_steps=2, the last 2 positions
    in recv_mask will be False (those observations are still in flight).
    """

    def __init__(self, maxlen: int, shape: tuple, dtype) -> None:
        """
        Parameters
        ----------
        maxlen : int    Maximum number of observations to retain (window size).
        shape  : tuple  Shape of a single observation (e.g. (4,)).
        dtype         : Numpy dtype (e.g. np.float32).
        """
        self.maxlen = maxlen
        self.shape = shape
        self.dtype = dtype

        self.buffer: np.ndarray = np.zeros((maxlen, *shape), dtype=dtype)
        self.recv: np.ndarray = np.zeros((maxlen,), dtype=bool)
        self.step_map: np.ndarray = np.full((maxlen,), -1, dtype=int)  # step for each slot
        self.current_step: int = -1  # -1 means buffer not started

    def add(self, obs, step: int) -> None:
        """
        Add an observation that arrived at a specific time step.

        Each call advances time; the buffer automatically tracks which time
        slots map to which buffer indices.

        Parameters
        ----------
        obs : np.ndarray or None
            If None, step has no observation (packet lost/delayed).
            If an ndarray, it is stored and recv[step_slot] = True.
        step : int
            The time step this observation belongs to.
        """
        # Update current step
        self.current_step = max(self.current_step, step)

        # Find which slot this step maps to
        # Slots represent window [current_step - maxlen + 1, ..., current_step]
        if step < 0:
            # Before buffer starts, ignore
            return

        # Calculate offset in window
        slot_idx = step % self.maxlen

        # Store observation
        if obs is not None:
            self.buffer[slot_idx] = obs
            self.recv[slot_idx] = True
        else:
            self.buffer[slot_idx] = 0
            self.recv[slot_idx] = False

        self.step_map[slot_idx] = step

    def get(self) -> tuple:
        """
        Return observations in the current time window in chronological order.

        Returns only observations that fit in the time window
        [current_step - maxlen + 1, ..., current_step]. Early observations
        before buffer initialization are excluded.

        Returns
        -------
        observations : np.ndarray, shape (num_steps, *shape)
        recv_mask    : np.ndarray, shape (num_steps,), dtype bool

        Raises
        ------
        ValueError if buffer is empty (no steps added yet).
        """
        if self.current_step < 0:
            raise ValueError("Buffer is empty")

        # Time window
        start_step = max(0, self.current_step - self.maxlen + 1)
        num_steps = self.current_step - start_step + 1

        obs_out = np.zeros((num_steps, *self.shape), dtype=self.dtype)
        mask_out = np.zeros((num_steps,), dtype=bool)

        for i in range(num_steps):
            step = start_step + i
            slot_idx = step % self.maxlen
            # Only fill if this slot actually maps to this step
            if self.step_map[slot_idx] == step:
                obs_out[i] = self.buffer[slot_idx]
                mask_out[i] = self.recv[slot_idx]

        return obs_out, mask_out

    def get_padded(self) -> tuple:
        """
        Return exactly `maxlen` observations in time-slot order.

        Returns observations for time window [current_step - maxlen + 1, ..., current_step].
        recv_mask[i] = True if that time slot's observation has been received.
        The most recent observation is at index [-1].

        Before buffer is initialized or for steps less than maxlen, earlier bounds
        are zero-padded with recv_mask=False.

        Returns
        -------
        observations : np.ndarray, shape (maxlen, *shape)
            Observations ordered by time step.
        recv_mask    : np.ndarray, shape (maxlen,), dtype bool
            True if that time slot's observation has arrived.
        """
        obs_out = np.zeros((self.maxlen, *self.shape), dtype=self.dtype)
        mask_out = np.zeros((self.maxlen,), dtype=bool)

        if self.current_step < 0:
            # Buffer not started yet
            return obs_out, mask_out

        # Time window
        start_step = max(0, self.current_step - self.maxlen + 1)

        for i in range(self.maxlen):
            step = start_step + i
            slot_idx = step % self.maxlen

            # Check if this slot contains data for this step
            if self.step_map[slot_idx] == step:
                # This slot is valid for this time step
                obs_out[i] = self.buffer[slot_idx]
                mask_out[i] = self.recv[slot_idx]
            # else: leave as zero with recv=False

        return obs_out, mask_out

    def clear(self) -> None:
        """Reset the buffer to the empty state."""
        self.buffer[:] = 0
        self.recv[:] = False
        self.step_map[:] = -1
        self.current_step = -1

    def __len__(self) -> int:
        """Return the number of time steps covered by the buffer."""
        if self.current_step < 0:
            return 0
        return min(self.current_step + 1, self.maxlen)

    @property
    def is_full(self) -> bool:
        """Return True if the buffer has filled with maxlen observations."""
        if self.current_step < 0:
            return False
        return (self.current_step + 1) >= self.maxlen
