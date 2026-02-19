from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch 
import numpy as np 


class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RL agent with configuration
        Args:
            config: Dictionary containing agent configurations
        
        """
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )


    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action given observation
        Args:
            observation: Current observation from environment
            deterministic: If True, select deterministic action (for evaluation)
        
        Returns:
            Tuple of (action, info_dict)
            - action: Selected action
            - info_dict: Additional information (log_prob, value, etc.)
        """
        pass 



    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            batch: Dictionary containing training data
        
        Returns:
            Dictionary of training metrics (loss, etc.)
        """
        pass



    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent to disk
        
        Args:
            path: Path to save checkpoint
        """
        pass




    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent from disk
        
        Args:
            path: Path to load checkpoint from
        """
        pass




    def train_mode(self) -> None:
        """Set agent to training mode"""
        self.network.train()
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode"""
        self.network.eval()
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()

    












