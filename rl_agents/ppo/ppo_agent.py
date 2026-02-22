import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union

from ..base_agent import BaseRLAgent
from ..networks import create_actor_critic_from_config
from ..buffers import RolloutBuffer
from ..distributions import make_action_distribution


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization (PPO) Agent
    
    Implements the clipped surrogate objective PPO algorithm with:
    - Learning rate scheduling support
    - TanhNormal distribution for bounded continuous actions
    - Numerically stable log probability computation
    - Support for both discrete and continuous action spaces
    - Vectorized environment support
    
    Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    
    Args:
        config: Configuration dictionary with keys:
            Network:
                - state_dim: State space dimension
                - action_dim: Action space dimension
                - is_discrete: Whether action space is discrete
                - network: Network configuration dict with:
                    - network_type: 'shared' or 'separate' (default: 'shared')
                    - observation_type: 'vector' or 'image' (default: 'vector')
                    - feature_dim: Feature dimension for shared networks (default: 64)
                    - hidden_dims: List of hidden layer sizes (default: [64, 64])
                    - activation: Activation function name (default: 'relu')
            
            PPO Hyperparameters:
                - learning_rate: Initial learning rate (default: 3e-4)
                - gamma: Discount factor (default: 0.99)
                - gae_lambda: GAE lambda parameter (default: 0.95)
                - clip_epsilon: PPO clipping parameter (default: 0.2)
                - n_epochs: Number of epochs per update (default: 10)
                - batch_size: Minibatch size (default: 64)
                - n_steps: Steps to collect before update (default: 2048)
                - value_loss_coef: Value loss coefficient (default: 0.5)
                - entropy_coef: Entropy bonus coefficient (default: 0.01)
                - max_grad_norm: Gradient clipping threshold (default: 0.5)
                - num_envs: Number of parallel environments (default: 1)
                - device: 'cpu', 'cuda', or 'auto' (default: 'cpu')
            
            Continuous Action Bounds (optional):
                - action_low: Lower bound (scalar or array)
                - action_high: Upper bound (scalar or array)
                  If provided, uses TanhNormal for bounded actions
            
            Learning Rate Scheduling (optional):
                - lr_scheduler: Scheduler type ('linear', 'cosine', 'step', 
                               'exponential', or None)
                - total_timesteps: Total training timesteps (required for 
                                  'linear' and 'cosine')
                - lr_step_size: Step size for 'step' scheduler (default: 10)
                - lr_gamma: Decay factor for 'step'/'exponential' (default: 0.9)
    
    Example:
        >>> # Discrete actions (CartPole)
        >>> config = {
        ...     'state_dim': 4,
        ...     'action_dim': 2,
        ...     'is_discrete': True,
        ...     'network': {
        ...         'network_type': 'shared',
        ...         'observation_type': 'vector',
        ...         'feature_dim': 64,
        ...         'hidden_dims': [64, 64],
        ...         'activation': 'tanh'
        ...     },
        ...     'learning_rate': 3e-4,
        ...     'lr_scheduler': 'linear',
        ...     'total_timesteps': 100000
        ... }
        >>> agent = PPOAgent(config)
        >>> 
        >>> # Continuous bounded actions (MuJoCo)
        >>> config_continuous = {
        ...     'state_dim': 17,
        ...     'action_dim': 6,
        ...     'is_discrete': False,
        ...     'action_low': -1.0,
        ...     'action_high': 1.0,
        ...     'network': {
        ...         'network_type': 'shared',
        ...         'observation_type': 'vector',
        ...         'feature_dim': 256,
        ...         'hidden_dims': [256, 256],
        ...         'activation': 'relu'
        ...     },
        ...     'learning_rate': 3e-4
        ... }
        >>> agent = PPOAgent(config_continuous)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.is_discrete = config['is_discrete']
        
        # PPO hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.n_steps = config.get('n_steps', 2048)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Action bounds for continuous actions (for TanhNormal)
        if not self.is_discrete:
            self.action_low = config.get('action_low', None)
            self.action_high = config.get('action_high', None)
        else:
            self.action_low = None
            self.action_high = None
        
        # Create network
        self.network = create_actor_critic_from_config(
            config=config['network'],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            is_discrete=self.is_discrete
        )
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )
        
        # Learning rate scheduler
        self._setup_lr_scheduler(config)
        
        # Rollout buffer
        num_envs = config.get('num_envs', 1)
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            num_envs=num_envs,
            observation_shape=(self.state_dim,),
            action_dim=1 if self.is_discrete else self.action_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Training mode by default
        self.network.train()
    
    def _setup_lr_scheduler(self, config: Dict[str, Any]) -> None:
        """
        Setup learning rate scheduler based on configuration
        
        Args:
            config: Configuration dictionary
        """
        scheduler_type = config.get('lr_scheduler', None)
        
        if scheduler_type is None or scheduler_type == 'none':
            # No learning rate scheduling
            self.scheduler = None
            
        elif scheduler_type == 'linear':
            # Linear decay from initial LR to 0
            total_timesteps = config.get('total_timesteps')
            if total_timesteps is None:
                raise ValueError(
                    "total_timesteps must be specified for linear LR scheduler"
                )
            
            total_updates = total_timesteps // self.n_steps
            
            def lr_lambda(update):
                """Linear decay"""
                return max(0.0, 1.0 - update / total_updates)
            
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda
            )
            
        elif scheduler_type == 'cosine':
            # Cosine annealing from initial LR to 0
            total_timesteps = config.get('total_timesteps')
            if total_timesteps is None:
                raise ValueError(
                    "total_timesteps must be specified for cosine LR scheduler"
                )
            
            total_updates = total_timesteps // self.n_steps
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_updates,
                eta_min=0.0
            )
            
        elif scheduler_type == 'step':
            # Step decay: multiply LR by gamma every step_size updates
            step_size = config.get('lr_step_size', 10)
            gamma = config.get('lr_gamma', 0.9)
            
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            
        elif scheduler_type == 'exponential':
            # Exponential decay: multiply LR by gamma every update
            gamma = config.get('lr_gamma', 0.99)
            
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            
        else:
            raise ValueError(
                f"Unknown lr_scheduler: '{scheduler_type}'. "
                f"Supported: 'linear', 'cosine', 'step', 'exponential', None"
            )
    
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action given observation
        
        Args:
            observation: Current observation
                - Single env: [state_dim] or [1, state_dim]
                - Vectorized: [num_envs, state_dim]
            deterministic: If True, select deterministic action (for evaluation)
        
        Returns:
            Tuple of (action, info_dict):
                - action: Selected action(s)
                    - Discrete: scalar or [num_envs]
                    - Continuous: [action_dim] or [num_envs, action_dim]
                - info_dict: Dictionary containing:
                    - 'value': Value estimate(s)
                    - 'log_prob': Log probability of action(s)
        
        Example:
            >>> obs = env.reset()[0]
            >>> action, info = agent.select_action(obs)
            >>> next_obs, reward, done, truncated, _ = env.step(action[0])
        """
        # Convert to tensor
        obs = torch.FloatTensor(observation).to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # [1, state_dim]
        
        with torch.no_grad():
            # Forward pass
            actor_output, value = self.network(obs)
            
            # Create distribution using factory
            dist = make_action_distribution(
                actor_output,
                is_discrete=self.is_discrete,
                action_low=self.action_low,
                action_high=self.action_high
            )
            
            if self.is_discrete:
                # Discrete actions (Categorical distribution)
                if deterministic:
                    probs = torch.softmax(actor_output, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action)  # [batch_size]
                
            else:
                # Continuous actions (TanhNormal or Normal)
                if deterministic:
                    # Use mean for deterministic action
                    action = dist.mean
                    
                    # Compute log_prob with numerical stability
                    if hasattr(dist, 'log_prob_from_u'):
                        # TanhNormal: use stable log_prob_from_u with base mean
                        # This avoids atanh for better numerical stability
                        u = dist.base.mean  # Mean of underlying Gaussian
                        log_prob = dist.log_prob_from_u(u).sum(dim=-1)  # [batch_size]
                    else:
                        # Normal: standard log_prob
                        log_prob = dist.log_prob(action).sum(dim=-1)  # [batch_size]
                else:
                    # Stochastic action (sampling)
                    if hasattr(dist, 'log_prob_from_u'):
                        # TanhNormal: returns (action, raw_u)
                        action, raw_u = dist.rsample()
                        # Use stable log_prob_from_u
                        log_prob = dist.log_prob_from_u(raw_u).sum(dim=-1)  # [batch_size]
                    else:
                        # Normal: returns action only
                        action = dist.rsample()
                        log_prob = dist.log_prob(action).sum(dim=-1)  # [batch_size]
            
            # Convert to numpy
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()
        
        info = {
            'value': value_np,
            'log_prob': log_prob_np
        }
        
        return action_np, info
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step (one minibatch update)
        
        Args:
            batch: Dictionary containing:
                - observations: [batch_size, state_dim]
                - actions: [batch_size, action_dim]
                - values: [batch_size] (old values from rollout)
                - log_probs: [batch_size] (old log probs from rollout)
                - advantages: [batch_size]
                - returns: [batch_size]
        
        Returns:
            Dictionary of training metrics:
                - loss: Total loss
                - policy_loss: Policy (actor) loss
                - value_loss: Value (critic) loss
                - entropy: Mean entropy (for exploration)
                - approx_kl: Approximate KL divergence
                - clip_fraction: Fraction of samples clipped
        """
        # Unpack batch
        obs = batch['observations']
        actions = batch['actions']
        old_values = batch['values']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        actor_output, values = self.network(obs)
        
        # Create distribution using factory
        dist = make_action_distribution(
            actor_output,
            is_discrete=self.is_discrete,
            action_low=self.action_low,
            action_high=self.action_high
        )
        
        if self.is_discrete:
            # Discrete actions
            actions_int = actions.long().squeeze(-1)
            log_probs = dist.log_prob(actions_int)  # [batch_size]
            entropy = dist.entropy()  # [batch_size]
            
        else:
            # Continuous actions
            # Compute log probability
            log_probs = dist.log_prob(actions)  # [batch_size, action_dim]
            # Sum over action dimensions if needed
            if log_probs.ndim > 1:
                log_probs = log_probs.sum(dim=-1)  # [batch_size]
            
            # Compute entropy
            entropy_per_dim = dist.entropy()  # [batch_size, action_dim]
            entropy = entropy_per_dim.sum(dim=-1)  # [batch_size]
        
        # Ratio: π_new(a|s) / π_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss (PPO's key innovation)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss (clipped for stability)
        values = values.squeeze(-1)
        value_loss_unclipped = (values - returns) ** 2
        
        # Clip value function (optional, helps stability)
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_epsilon,
            self.clip_epsilon
        )
        value_loss_clipped = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients (prevents exploding gradients)
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute metrics for logging
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean().item(),
            'clip_fraction': (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()
        }
    
    def update(self, last_value: Union[float, np.ndarray] = 0.0) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout
        
        This is called after collecting n_steps of experience.
        It computes advantages using GAE, then trains the network
        for n_epochs using random minibatches.
        
        Args:
            last_value: Bootstrap value for last state(s)
                - Single env: scalar (0 if terminated)
                - Vectorized: [num_envs] array
        
        Returns:
            Dictionary of training metrics averaged over all epochs:
                - loss: Total loss
                - policy_loss: Policy loss
                - value_loss: Value loss
                - entropy: Mean entropy
                - approx_kl: Approximate KL divergence
                - clip_fraction: Fraction of clipped samples
                - learning_rate: Current learning rate
        
        Example:
            >>> # Collect rollout
            >>> for step in range(n_steps):
            ...     action, info = agent.select_action(obs)
            ...     next_obs, reward, done, truncated, _ = env.step(action[0])
            ...     agent.buffer.add(obs, action, reward, info['value'], 
            ...                      info['log_prob'], done)
            ...     obs = next_obs
            >>> 
            >>> # Update agent
            >>> last_val = 0.0 if done else agent.select_action(obs)[1]['value'][0]
            >>> metrics = agent.update(last_val)
        """
        # Compute returns and advantages using GAE
        self.buffer.compute_returns_and_advantages(last_value)
        
        # Train for multiple epochs
        all_metrics = []
        
        for epoch in range(self.n_epochs):
            # Sample random minibatches
            for batch in self.buffer.get(self.batch_size):
                metrics = self.train_step(batch)
                all_metrics.append(metrics)
        
        # Average metrics across all minibatches and epochs
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            # Record current learning rate
            avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        else:
            # Record static learning rate
            avg_metrics['learning_rate'] = self.learning_rate
        
        # Clear buffer for next rollout
        self.buffer.clear()
        
        return avg_metrics
    
    def save(self, path: str) -> None:
        """
        Save agent checkpoint to disk
        
        Args:
            path: Path to save checkpoint (e.g., 'checkpoints/ppo_cartpole.pt')
        
        Example:
            >>> agent.save('checkpoints/model_100k.pt')
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """
        Load agent checkpoint from disk
        
        Args:
            path: Path to load checkpoint from
        
        Example:
            >>> agent = PPOAgent(config)
            >>> agent.load('checkpoints/model_100k.pt')
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.network.to(self.device)
    
    def train_mode(self) -> None:
        """Set agent to training mode (enables dropout, batchnorm, etc.)"""
        self.network.train()
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode (disables dropout, batchnorm, etc.)"""
        self.network.eval()
    
    def get_current_lr(self) -> float:
        """
        Get current learning rate
        
        Returns:
            Current learning rate (float)
        
        Example:
            >>> lr = agent.get_current_lr()
            >>> print(f"Current LR: {lr:.6f}")
        """
        return self.optimizer.param_groups[0]['lr']