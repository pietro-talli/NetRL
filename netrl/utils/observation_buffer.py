import numpy as np


class ObservationBuffer:
    """
    Circular buffer for storing RL observations with loss tracking.

    Slots can hold a real observation (recv=True) or be marked as missing
    (recv=False) when a packet was lost or delayed beyond the current step.
    """

    def __init__(self, maxlen: int, shape: tuple, dtype) -> None:
        """
        Parameters
        ----------
        maxlen : int    Maximum number of observations to retain.
        shape  : tuple  Shape of a single observation (e.g. (4,)).
        dtype         : Numpy dtype (e.g. np.float32).
        """
        self.maxlen = maxlen
        self.shape = shape
        self.dtype = dtype

        self.buffer: np.ndarray = np.zeros((maxlen, *shape), dtype=dtype)
        self.recv: np.ndarray = np.zeros((maxlen,), dtype=bool)
        self.index: int = 0   # write pointer (next slot to fill)
        self.size: int = 0    # number of valid entries (0..maxlen)

    def add(self, obs) -> None:
        """
        Push one entry into the buffer.

        Parameters
        ----------
        obs : np.ndarray or None
            If None, the slot is zeroed and recv[slot] = False.
            If an ndarray, it is stored and recv[slot] = True.
        """
        if obs is not None:
            self.buffer[self.index] = obs
            self.recv[self.index] = True
        else:
            self.buffer[self.index] = 0
            self.recv[self.index] = False
        self.index = (self.index + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def get(self) -> tuple:
        """
        Return the `size` most recent entries in chronological order.

        Returns
        -------
        observations : np.ndarray, shape (size, *shape)
        recv_mask    : np.ndarray, shape (size,), dtype bool

        Raises
        ------
        ValueError if the buffer is empty.
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")
        idx = (self.index - self.size + np.arange(self.size)) % self.maxlen
        return self.buffer[idx].copy(), self.recv[idx].copy()

    def get_padded(self) -> tuple:
        """
        Always return exactly `maxlen` entries in chronological order.

        Slots that have never been written are returned as zeros with
        recv_mask = False. The most recent entry is at index [-1].
        This method produces a shape compatible with a static
        gymnasium.spaces.Box defined on the wrapper.

        Returns
        -------
        observations : np.ndarray, shape (maxlen, *shape)
        recv_mask    : np.ndarray, shape (maxlen,), dtype bool
        """
        obs_out = np.zeros((self.maxlen, *self.shape), dtype=self.dtype)
        mask_out = np.zeros((self.maxlen,), dtype=bool)
        if self.size > 0:
            idx = (self.index - self.size + np.arange(self.size)) % self.maxlen
            obs_out[-self.size:] = self.buffer[idx]
            mask_out[-self.size:] = self.recv[idx]
        return obs_out, mask_out

    def clear(self) -> None:
        """Reset the buffer to the empty state."""
        self.buffer[:] = 0
        self.recv[:] = False
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size == self.maxlen
