import torch 
import torch.nn.functional as F 
from torch.distributions import Normal, Categorical
import math 
from typing import Union, Tuple, Optional




class TanhNormal:
    """
    A Tanh-transformed Normal distribution for bounded continuous actions.
    
    This distribution is more numerically stable than applying tanh squashing
    after sampling from a normal distribution, because it avoids the atanh
    operation in log_prob calculation.
    
    The action space is transformed from unbounded Gaussian to [low, high].
    
    Args:
        mu: Mean of the underlying Gaussian [batch_size, action_dim]
        std: Standard deviation of the underlying Gaussian [batch_size, action_dim]
        low: Lower bound of action space (scalar or tensor)
        high: Upper bound of action space (scalar or tensor)
    
    Reference:
        - "Soft Actor-Critic" (Haarnoja et al., 2018)
        - Stable Baselines3 implementation
    
    Example:
        >>> mu = torch.zeros(32, 6)
        >>> std = torch.ones(32, 6)
        >>> dist = TanhNormal(mu, std, low=-1.0, high=1.0)
        >>> action, raw_u = dist.rsample()  # action in [-1, 1]
        >>> log_prob = dist.log_prob_from_u(raw_u)
    """
    
    def __init__(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
        low: Optional[Union[torch.Tensor, float]] = None,
        high: Optional[Union[torch.Tensor, float]] = None
    ):
        self.device = mu.device
        self.dtype = mu.dtype
        
        # Underlying Gaussian distribution
        self.base = Normal(mu, std)
        
        # Compute scale and bias for transforming tanh output to [low, high]
        if low is None or high is None:
            # No bounds: output is tanh(u) in [-1, 1]
            self._scale = 1.0
            self._bias = 0.0
        else:
            # Transform tanh(u) from [-1, 1] to [low, high]
            # a = bias + scale * tanh(u)
            # where scale = (high - low) / 2, bias = (high + low) / 2
            low_t = torch.as_tensor(low, device=self.device, dtype=self.dtype)
            high_t = torch.as_tensor(high, device=self.device, dtype=self.dtype)
            self._scale = (high_t - low_t) / 2.0
            self._bias = (high_t + low_t) / 2.0
    
    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        """
        Apply tanh squashing and scale to [low, high]
        
        Args:
            u: Raw Gaussian samples [batch_size, action_dim]
        
        Returns:
            Squashed actions in [low, high]
        """
        return self._bias + self._scale * torch.tanh(u)
    
    def _unsquash(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of squashing (used for log_prob from action)
        
        Args:
            a: Actions in [low, high]
        
        Returns:
            Tuple of (u, z) where:
                - u: Raw Gaussian samples
                - z: Intermediate value in [-1, 1]
        """
        # Map [low, high] → [-1, 1]
        z = (a - self._bias) / (self._scale + 1e-8)
        
        # Numerical safety: clip to avoid infinity at ±1.0
        z = torch.clamp(z, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Apply atanh to get raw Gaussian samples
        u = torch.atanh(z)
        
        return u, z
    
    def rsample(
        self,
        sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sampling (differentiable)
        
        Args:
            sample_shape: Shape of samples to draw
        
        Returns:
            Tuple of (squashed_action, raw_u)
        """
        u = self.base.rsample(sample_shape)
        return self._squash(u), u
    
    def sample(
        self,
        sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-reparameterized sampling
        
        Args:
            sample_shape: Shape of samples to draw
        
        Returns:
            Tuple of (squashed_action, raw_u)
        """
        u = self.base.sample(sample_shape)
        return self._squash(u), u
    
    def log_prob_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probability from raw Gaussian samples (STABLE)
        
        This is much more numerically stable than calculating from
        the squashed action 'a' because it avoids atanh.
        
        Args:
            u: Raw Gaussian samples [batch_size, action_dim]
        
        Returns:
            Log probability [batch_size] (summed over action_dim)
        """
        # Log probability under Gaussian
        logp_u = self.base.log_prob(u)  # [batch_size, action_dim]
        
        # Change of variables correction: log|da/du|
        # a = bias + scale * tanh(u)
        # da/du = scale * (1 - tanh²(u))
        # log|da/du| = log(scale) + log(1 - tanh²(u))
        
        # Stable computation of log(1 - tanh²(u))
        # Using: log(1 - tanh²(u)) = 2 * (log(2) - u - softplus(-2u))
        tanh_correction = 2.0 * (
            math.log(2.0) - u - F.softplus(-2.0 * u)
        )  # [batch_size, action_dim]
        
        # Total log probability (change of variables)
        # log p(a) = log p(u) - log|da/du|
        log_prob = (
            logp_u 
            - torch.log(torch.as_tensor(self._scale + 1e-8, device=self.device))
            - tanh_correction
        )  # [batch_size, action_dim]
        
        # Sum over action dimensions
        return log_prob.sum(dim=-1)  # [batch_size]
    
    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """
        Standard log probability (less stable than log_prob_from_u)
        
        Use log_prob_from_u when possible for better numerical stability.
        
        Args:
            a: Actions in [low, high] [batch_size, action_dim]
        
        Returns:
            Log probability [batch_size]
        """
        u, _ = self._unsquash(a)
        return self.log_prob_from_u(u)
    
    def entropy(self) -> torch.Tensor:
        """
        Entropy of the underlying Gaussian distribution
        
        Note: This is an approximation, as the true entropy of
        the transformed distribution is intractable.
        
        Returns:
            Entropy [batch_size, action_dim]
        """
        return self.base.entropy()
    
    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution (deterministic squashed mean)"""
        return self._squash(self.base.mean)
    
    @property
    def stddev(self) -> torch.Tensor:
        """Standard deviation of underlying Gaussian"""
        return self.base.stddev


def make_action_distribution(
    policy_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    is_discrete: bool,
    action_low: Optional[Union[torch.Tensor, float]] = None,
    action_high: Optional[Union[torch.Tensor, float]] = None
):
    """
    Factory function to create action distribution from policy output
    
    Args:
        policy_output: 
            - For discrete: logits [batch_size, num_actions]
            - For continuous: tuple of (mean, log_std)
        is_discrete: Whether action space is discrete
        action_low: Lower bound for continuous actions (None for discrete)
        action_high: Upper bound for continuous actions (None for discrete)
    
    Returns:
        Distribution object (Categorical or TanhNormal or Normal)
    
    Examples:
        >>> # Discrete actions
        >>> logits = torch.randn(32, 4)
        >>> dist = make_action_distribution(logits, is_discrete=True)
        >>> action = dist.sample()
        >>> 
        >>> # Continuous unbounded
        >>> mean = torch.zeros(32, 6)
        >>> log_std = torch.zeros(32, 6)
        >>> dist = make_action_distribution((mean, log_std), is_discrete=False)
        >>> 
        >>> # Continuous bounded
        >>> dist = make_action_distribution(
        ...     (mean, log_std),
        ...     is_discrete=False,
        ...     action_low=-1.0,
        ...     action_high=1.0
        ... )
    """
    if is_discrete:
        # Categorical expects logits (unnormalized log probabilities)
        return Categorical(logits=policy_output)
    
    else:
        # Continuous actions
        mean, log_std = policy_output
        std = torch.exp(log_std)
        
        # If bounds are provided, use stable TanhNormal
        if action_low is not None and action_high is not None:
            return TanhNormal(mean, std, action_low, action_high)
        
        # Otherwise, unbounded Normal distribution
        return Normal(mean, std)
    


