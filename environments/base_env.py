from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym 
import numpy as np 



class BaseEnvironment(ABC):
    """Abstract base class for all RL environments"""
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize environment with configuration
        Args:
            config: Dictionary containing environment parameters
        """
        self.config = config 
        self.env = None 

    @abstractmethod
    def make_env(self) -> gym.Env:
        """Create and return the gymnasium environment"""
        pass 

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return state dimension"""
        pass 

    @abstractmethod
    def get_action_dim(self) -> int:
        """Return action dimension"""
        pass 


    @abstractmethod
    def is_discrete(self) -> bool:
        """Return True if action space is discrete"""
        pass 


    def get_action_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return action bounds for continuous action spaces
        """
        return None
    
    
    def reset(self):
        """Reset environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call make_env() first.")
        return self.env.reset()
    

    def step(self, action):
        """Take a step in the environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        return self.env.step(action)
    

    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()


