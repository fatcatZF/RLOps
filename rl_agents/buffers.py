
import torch 
import numpy as np
from typing import Union, Dict, Tuple, Optional, Generator






class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, A2C, TRPO)
    Stores trajectories and computes GAE

    Args:
        buffer_size: Number of steps to store PER environment
        num_envs: Number of parallel environments (default: 1)
        observation_shape: Shape of single observation (e.g., (4,) for CartPole)
        action_dim: Dimension of action space (1 for discrete stored as scalar)
        device: Device to store tensors on ('cpu' or 'cuda')
        gamma: Discount factor for rewards (default: 0.99)
        gae_lambda: GAE lambda parameter for bias-variance tradeoff (default: 0.95)
    
    Storage shape: [buffer_size, num_envs, ...]
    - For single env (num_envs=1): effectively [buffer_size, 1, ...]
    - For vectorized (num_envs=N): [buffer_size, N, ...]

    Examples:
        >>> # Single environment (CartPole)
        >>> buffer = RolloutBuffer(
        ...     buffer_size=2048,
        ...     num_envs=1,
        ...     observation_shape=(4,),
        ...     action_dim=1,
        ...     gamma=0.99,
        ...     gae_lambda=0.95
        ... )
        >>> 
        >>> # 8 parallel environments (8x faster data collection)
        >>> buffer = RolloutBuffer(
        ...     buffer_size=256,   # 256 steps * 8 envs = 2048 total experiences
        ...     num_envs=8,
        ...     observation_shape=(4,),
        ...     action_dim=1
        ... )
    
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int = 1,
        observation_shape: Tuple[int, ...] = (4,),
        action_dim: int = 1,
        device: str = 'cpu',
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Current position in buffer
        self.pos = 0
        self.full = False

        # Storage arrays: [buffer_size, num_envs, ...]
        # Using numpy for memory efficiency, convert to torch only when sampling
        self.observations = np.zeros(
            (buffer_size, num_envs) + observation_shape,
            dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, num_envs, action_dim),
            dtype=np.float32
        )


        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)

        # Computed by compute_returns_and_advantages()
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)


        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
        done: Union[bool, np.ndarray]
    ) -> None:
        """
        Add one step of experience from all environments
        
        Args:
            obs: Observations from environments
                - Single env: [*observation_shape] or [1, *observation_shape]
                - Vectorized: [num_envs, *observation_shape]
            action: Actions taken
                - Single env: [action_dim] or scalar or [1, action_dim]
                - Vectorized: [num_envs, action_dim]
            reward: Rewards received
                - Single env: scalar or [1]
                - Vectorized: [num_envs]
            value: Value estimates
                - Single env: scalar or [1]
                - Vectorized: [num_envs]
            log_prob: Log probabilities of actions
                - Single env: scalar or [1]
                - Vectorized: [num_envs]
            done: Done flags
                - Single env: bool or [1]
                - Vectorized: [num_envs]
        
        Examples:
            >>> # Single environment
            >>> buffer.add(
            ...     obs=np.array([0.1, 0.2, 0.3, 0.4]),  # [4]
            ...     action=np.array([1]),                # [1]
            ...     reward=1.0,                          # scalar
            ...     value=0.5,                           # scalar
            ...     log_prob=-0.69,                      # scalar
            ...     done=False                           # bool
            ... )
            >>> 
            >>> # 8 parallel environments
            >>> buffer.add(
            ...     obs=np.array([[...], ..., [...]]),  # [8, 4]
            ...     action=np.array([[1], ..., [0]]),   # [8, 1]
            ...     reward=np.array([1.0, ..., 0.0]),   # [8]
            ...     value=np.array([0.5, ..., 0.3]),    # [8]
            ...     log_prob=np.array([-0.69, ...]),    # [8]
            ...     done=np.array([False, ..., True])   # [8]
            ... )
        """
        
        if self.pos >= self.buffer_size:
            raise RuntimeError(
                f"Buffer overflow: trying to add to position {self.pos} "
                f"but buffer_size is {self.buffer_size}"
            )
        
        # Convert to numpy arrays
        obs = np.array(obs)
        action = np.array(action)
        reward = np.array(reward)
        value = np.array(value)
        log_prob = np.array(log_prob)
        done = np.array(done)



        # Reshape to [num_envs, ...] if necessary
        if self.num_envs == 1:
            # Single environment: ensure shape is [1, ...]
            if obs.shape == self.observation_shape:
                obs = obs.reshape((1,) + self.observation_shape)
            
            if action.shape == (self.action_dim,):
                action = action.reshape(1, self.action_dim)
            elif action.ndim == 0:  # Scalar
                action = np.array([[action]])
            elif action.shape == (1,):
                action = action.reshape(1, 1)
            
            if reward.ndim == 0:  # Scalar
                reward = np.array([reward])
            
            if value.ndim == 0:  # Scalar
                value = np.array([value])
            
            if log_prob.ndim == 0:  # Scalar
                log_prob = np.array([log_prob])
            
            if done.ndim == 0:  # Bool/scalar
                done = np.array([float(done)])

        

        # Validate shapes
        assert obs.shape == (self.num_envs,) + self.observation_shape, \
            f"Expected obs shape {(self.num_envs,) + self.observation_shape}, got {obs.shape}"
        assert action.shape == (self.num_envs, self.action_dim), \
            f"Expected action shape {(self.num_envs, self.action_dim)}, got {action.shape}"
        assert reward.shape == (self.num_envs,), \
            f"Expected reward shape {(self.num_envs,)}, got {reward.shape}"
        assert value.shape == (self.num_envs,), \
            f"Expected value shape {(self.num_envs,)}, got {value.shape}"
        assert log_prob.shape == (self.num_envs,), \
            f"Expected log_prob shape {(self.num_envs,)}, got {log_prob.shape}"
        assert done.shape == (self.num_envs,), \
            f"Expected done shape {(self.num_envs,)}, got {done.shape}"
        

        # Store
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    
    def compute_returns_and_advantages(
            self,
            last_values: Union[float, np.ndarray]
    ) -> None:
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation)
        
        Must be called after collecting a full rollout and before sampling.
        
        Args:
            last_values: Value estimates for the last states (bootstrap values)
                - Single env: scalar or [1]
                - Vectorized: [num_envs]
                Set to 0 if episode terminated, otherwise use V(s_last)
        
        GAE Algorithm:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)
        
        Where:
            - δ_t: TD error
            - A_t: Advantage estimate
            - R_t: Return estimate
            - γ: Discount factor (gamma)
            - λ: GAE lambda parameter
        
        Example:
            >>> # Single environment
            >>> last_value = 0.0 if done else agent.get_value(last_obs)
            >>> buffer.compute_returns_and_advantages(last_value)
            >>> 
            >>> # Vectorized environments
            >>> last_values = np.zeros(8)  # All done
            >>> # OR
            >>> last_values = agent.get_value(last_obs)  # [8]
            >>> buffer.compute_returns_and_advantages(last_values)
        """

        # Convert to numpy array and reshape
        last_values = np.array(last_values)
        if last_values.ndim == 0:
            last_values = np.array([last_values])
        
        assert last_values.shape == (self.num_envs,), \
            f"Expected last_values shape {(self.num_envs,)}, got {last_values.shape}"
        
        # Initialize advantages for each environment
        last_gae_lambda = np.zeros(self.num_envs, dtype=np.float32)


        # Compute GAE by iterating backwards through time
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                # Last step: use provided last_values for bootstrap
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                # Use value from next step
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            # Shape: [num_envs]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            
            # GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
            # Shape: [num_envs]
            last_gae_lambda = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )
            
            self.advantages[step] = last_gae_lambda


        # Returns: R_t = A_t + V(s_t)
        # Shape: [buffer_size, num_envs]
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]


    
    def get_minibatch(
        self,
        batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate random minibatches for training
        
        Flattens the buffer from [buffer_size, num_envs, ...] to [buffer_size * num_envs, ...]
        then yields random minibatches (without replacement).
        
        Args:
            batch_size: Size of each minibatch
        
        Yields:
            Dictionary containing:
                - observations: [batch_size, *observation_shape]
                - actions: [batch_size, action_dim]
                - values: [batch_size]
                - log_probs: [batch_size]
                - advantages: [batch_size]
                - returns: [batch_size]
        
        Example:
            >>> # Train for multiple epochs
            >>> for epoch in range(10):
            ...     for batch in buffer.get(batch_size=64):
            ...         loss = agent.train_step(batch)
        """
        # Total number of experiences
        total_steps = self.pos * self.num_envs
        
        # Flatten [buffer_size, num_envs, ...] → [total_steps, ...]
        obs_flat = self.observations[:self.pos].reshape(
            (total_steps,) + self.observation_shape
        )
        actions_flat = self.actions[:self.pos].reshape(
            total_steps, self.action_dim
        )
        values_flat = self.values[:self.pos].reshape(total_steps)
        log_probs_flat = self.log_probs[:self.pos].reshape(total_steps)
        advantages_flat = self.advantages[:self.pos].reshape(total_steps)
        returns_flat = self.returns[:self.pos].reshape(total_steps)
        
        # Generate random permutation of indices
        indices = np.arange(total_steps)
        np.random.shuffle(indices)
        
        # Yield minibatches
        start_idx = 0
        while start_idx < total_steps:
            # Get batch indices
            end_idx = min(start_idx + batch_size, total_steps)
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch dictionary and convert to torch tensors
            yield {
                'observations': torch.as_tensor(
                    obs_flat[batch_indices], 
                    dtype=torch.float32
                ).to(self.device),
                'actions': torch.as_tensor(
                    actions_flat[batch_indices],
                    dtype=torch.float32
                ).to(self.device),
                'values': torch.as_tensor(
                    values_flat[batch_indices],
                    dtype=torch.float32
                ).to(self.device),
                'log_probs': torch.as_tensor(
                    log_probs_flat[batch_indices],
                    dtype=torch.float32
                ).to(self.device),
                'advantages': torch.as_tensor(
                    advantages_flat[batch_indices],
                    dtype=torch.float32
                ).to(self.device),
                'returns': torch.as_tensor(
                    returns_flat[batch_indices],
                    dtype=torch.float32
                ).to(self.device),
            }
            
            start_idx = end_idx


    def clear(self) -> None:
        """
        Clear the buffer
        
        Resets position to 0, ready for next rollout.
        Call this after each PPO update.
        """
        self.pos = 0
        self.full = False


    
    def size(self) -> int:
        """
        Return total number of experiences stored
        
        Returns:
            Total experiences = (current_position * num_envs)
        """
        return (self.pos if not self.full else self.buffer_size) * self.num_envs 
    


    def __len__(self) -> int:
        """Return total number of experiences (same as size())"""
        return self.size()




    
    






    


















class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms (SAC, TD3, DQN)
    Stores transitions for experience replay
    """
    pass



















class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer (for Rainbow DQN, etc.)
    """
    pass