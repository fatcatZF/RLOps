from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import gymnasium as gym 
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np 


class BaseEnvironment(ABC):
    """
    Abstract base class for all RL environments
    
    Supports both single and vectorized environments (Sync and Async).
    The vectorization is handled automatically based on num_envs in config.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize environment with configuration
        
        Args:
            config: Dictionary containing environment parameters
                - num_envs: Number of parallel environments (default: 1)
                - vectorization_mode: 'sync' or 'async' (default: 'sync')
        """
        self.config = config 
        self.num_envs = config.get('num_envs', 1)
        self.vectorization_mode = config.get('vectorization_mode', 'sync')
        self.is_vectorized = self.num_envs > 1
        
        self.env = None 
        self._make_vectorized_env()

    @abstractmethod
    def make_env(self) -> gym.Env:
        """
        Create and return a single gymnasium environment
        
        This method must be implemented by subclasses to create
        one instance of the environment.
        """
        pass 

    def _make_vectorized_env(self):
        """
        Create vectorized environment or single environment
        
        This is called automatically during __init__.
        Creates either:
        - Single gym.Env if num_envs == 1
        - SyncVectorEnv if num_envs > 1 and mode == 'sync'
        - AsyncVectorEnv if num_envs > 1 and mode == 'async'
        """
        if self.is_vectorized:
            # Create list of environment factory functions
            env_fns = [self.make_env for _ in range(self.num_envs)]
            
            # Create vectorized environment
            if self.vectorization_mode == 'async':
                self.env = AsyncVectorEnv(env_fns)
            else:
                self.env = SyncVectorEnv(env_fns)
        else:
            # Create single environment
            self.env = self.make_env()

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
        
        Returns:
            Tuple of (low, high) arrays or None for discrete spaces
        """
        return None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment
        
        Args:
            seed: Random seed
            options: Additional reset options
        
        Returns:
            - Single env: (observation, info)
            - Vectorized: (observations, infos) with shape [num_envs, ...]
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        
        if seed is not None:
            if self.is_vectorized:
                # For vectorized env, create list of seeds
                seeds = [seed + i for i in range(self.num_envs)] if seed else None
                return self.env.reset(seed=seeds, options=options)
            else:
                return self.env.reset(seed=seed, options=options)
        else:
            return self.env.reset(options=options)
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action(s) to take
                - Single env: int or array
                - Vectorized: array of shape [num_envs] or [num_envs, action_dim]
        
        Returns:
            - Single env: (obs, reward, terminated, truncated, info)
            - Vectorized: (obs, rewards, terminated, truncated, infos)
                          with arrays of shape [num_envs, ...]
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        return self.env.step(action)
    
    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()
    
    def render(self):
        """
        Render the environment
        
        Note: Rendering only works for single environments
        """
        if self.env is not None and not self.is_vectorized:
            return self.env.render()
        elif self.is_vectorized:
            print("Warning: Rendering not supported for vectorized environments")
    
    def __repr__(self) -> str:
        """String representation"""
        env_type = f"Vectorized-{self.vectorization_mode.upper()}" if self.is_vectorized else "Single"
        return (f"{self.__class__.__name__}({env_type}, num_envs={self.num_envs}, "
                f"state_dim={self.get_state_dim()}, action_dim={self.get_action_dim()})")

