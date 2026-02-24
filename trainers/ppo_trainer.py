"""
PPO Trainers

Contains:
- PPOTrainer: Standard PPO training
- DistributedPPOTrainer: Distributed PPO with multi-GPU support
"""


import numpy as np
import torch
from typing import Dict, Any, Tuple, List

from .base_trainer import BaseTrainer
from .distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    get_world_info,
    synchronize_metrics,
    wrap_model_ddp
)





class PPOTrainer(BaseTrainer):
    """
    Standard PPO trainer
    
    Handles PPO-specific rollout collection and bootstrapping.
    
    Args:
        agent: PPO agent instance
        train_env: Training environment
        eval_env: Evaluation environment (optional)
        config: Training configuration dictionary
        experiment_name: Name for this experiment
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        verbose: Verbosity level
    
    Example:
        >>> from rl_agents.ppo import PPOAgent
        >>> from environments import create_environment
        >>> from trainers import PPOTrainer
        >>> from utils.experiments import load_config
        >>> 
        >>> config = load_config('config.yaml')
        >>> agent = PPOAgent(agent_config)
        >>> train_env = create_environment('cartpole', env_config)
        >>> 
        >>> trainer = PPOTrainer(
        ...     agent=agent,
        ...     train_env=train_env,
        ...     config=config,
        ...     experiment_name='ppo_cartpole'
        ... )
        >>> 
        >>> results = trainer.train()
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PPO trainer"""
        super().__init__(*args, **kwargs)
        
        # PPO-specific state tracking
        self.current_episode_rewards = np.zeros(self.num_envs)
        self.done = False  # For single env
        self.dones = np.zeros(self.num_envs, dtype=bool)  # For vectorized env
    
    def _collect_rollout(
        self,
        obs: np.ndarray,
        episode_rewards: List[float],
        n_steps: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Collect PPO rollout
        
        Args:
            obs: Current observation(s)
            episode_rewards: List of completed episode rewards
            n_steps: Number of steps to collect
        
        Returns:
            Tuple of (final_obs, updated_episode_rewards)
        """
        for step in range(n_steps):
            # Select action
            action, info = self.agent.select_action(obs)
            
            # Step environment
            if self.is_vectorized:
                next_obs, rewards, terminated, truncated, env_info = self.train_env.step(action)
                self.dones = np.logical_or(terminated, truncated)
                
                # Add to buffer
                self.agent.buffer.add(
                    obs=obs,
                    action=action,
                    reward=rewards,
                    value=info['value'],
                    log_prob=info['log_prob'],
                    done=self.dones,
                    raw_action=info['raw_action']
                )
                
                # Track episode rewards
                self.current_episode_rewards += rewards
                for i in range(self.num_envs):
                    if self.dones[i]:
                        episode_rewards.append(self.current_episode_rewards[i])
                        self.current_episode_rewards[i] = 0
                
                obs = next_obs
            else:
                next_obs, reward, terminated, truncated, env_info = self.train_env.step(action[0])
                self.done = terminated or truncated
                
                # Add to buffer
                self.agent.buffer.add(
                    obs=obs,
                    action=np.array([action[0]]),
                    reward=reward,
                    value=info['value'][0],
                    log_prob=info['log_prob'][0],
                    done=self.done,
                    raw_action=info['raw_action']
                )
                
                # Track episode rewards
                self.current_episode_rewards[0] += reward
                if self.done:
                    episode_rewards.append(self.current_episode_rewards[0])
                    self.current_episode_rewards[0] = 0
                    obs, _ = self.train_env.reset()
                else:
                    obs = next_obs
        
        return obs, episode_rewards
    
    def _update_agent(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Update PPO agent
        
        Args:
            obs: Current observation(s) for computing last values
        
        Returns:
            Dictionary of training metrics
        """
        # Compute last values for bootstrapping
        last_values = self._compute_last_values(obs)
        
        # Update agent
        metrics = self.agent.update(last_values)
        
        return metrics
    
    def _compute_last_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute last values for PPO bootstrapping
        
        Args:
            obs: Current observation(s)
        
        Returns:
            Last values for GAE computation
        """
        if self.is_vectorized:
            _, info = self.agent.select_action(obs)
            last_values = info['value']
            
            # Ensure correct shape
            if isinstance(last_values, np.ndarray) and last_values.ndim > 1:
                last_values = last_values.squeeze()
            
            # Set to 0 where episodes are done
            last_values = np.where(self.dones, 0.0, last_values)
        else:
            last_val = 0.0 if self.done else self.agent.select_action(obs)[1]['value'][0]
            last_values = last_val
        
        return last_values
    
    def _add_metrics_to_progress(self, progress: str, metrics: Dict[str, float]) -> str:
        """Add PPO-specific metrics to progress string"""
        progress = super()._add_metrics_to_progress(progress, metrics)
        
        if 'entropy' in metrics:
            progress += f" | Entropy: {metrics['entropy']:.4f}"
        if 'approx_kl' in metrics:
            progress += f" | KL: {metrics['approx_kl']:.4f}"
        if 'learning_rate' in metrics:
            progress += f" | LR: {metrics['learning_rate']:.6f}"
        
        return progress




























