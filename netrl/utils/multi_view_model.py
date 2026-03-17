from gymnasium import Env, spaces
import numpy as np
import gymnasium as gym

from typing import List

class MultiViewModel:
    def __init__(self, 
                 observer_ids: List[str],
                 obs_shapes: List[tuple],
                 obs_dtypes: List[np.dtype]
                 ):
        self.observer_ids = observer_ids
        self.obs_shapes = obs_shapes
        self.obs_dtypes = obs_dtypes

        self.spaces = {
            oid: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=shape,
                    dtype=dtype,
                )
            for oid, shape, dtype in zip(observer_ids, obs_shapes, obs_dtypes)
        }

    def observe(self, env: gym.Env = None, state = None):
        """
        Class to implement the observations of each node 
        """
        raise NotImplementedError("This method should be implemented by the user to return the observations for each observer.")