from gymnasium import spaces
import cv2
import numpy as np
import gymnasium as gym

class ImageEnvWrapper(gym.Wrapper):
    def __init__(self, env, height: int = 64, width: int = 64, channels: int = 3):
        super().__init__(env)
        # Override observation space to be an image (e.g. 64x64 RGB)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(height, width, channels),
            dtype=np.float32, # use float32 for normalized images
        )
        self.height = height
        self.width = width
        self.channels = channels

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        img_obs = self.get_img()
        return img_obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        img_obs = self.get_img()
        return img_obs, reward, term, trunc, info
    
    def get_img(self):
        img = self.env.render()
        # Resize to self.height, self.width
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return img.astype(np.float32) / 255.0  # normalize to [0, 1]