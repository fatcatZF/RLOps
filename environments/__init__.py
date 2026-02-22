from .base_env import BaseEnvironment
from .cartpole import CartPoleEnvironment


# Registry of available environments
ENVIRONMENTS = {
    'cartpole': CartPoleEnvironment,
    # Add more environments here
}


def create_environment(env_name: str, config: dict) -> BaseEnvironment:
    """
    Factory function to create environments
    
    Automatically handles single vs vectorized based on config['num_envs'].
    
    Args:
        env_name: Name of environment (e.g., 'cartpole')
        config: Configuration dictionary with:
            - num_envs: Number of parallel environments (default: 1)
            - vectorization_mode: 'sync' or 'async' (default: 'sync')
            - Other environment-specific parameters

    Returns:
        BaseEnvironment instance (single or vectorized)

    Raises:
        ValueError: If env_name not found in registry
    
    Examples:
        >>> # Single environment
        >>> env = create_environment('cartpole', {'num_envs': 1})
        >>> 
        >>> # Vectorized (sync)
        >>> env = create_environment('cartpole', {'num_envs': 8, 'vectorization_mode': 'sync'})
        >>> 
        >>> # Vectorized (async - better for CPU-intensive envs)
        >>> env = create_environment('cartpole', {'num_envs': 8, 'vectorization_mode': 'async'})
    """
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available: {list(ENVIRONMENTS.keys())}"
        )
    
    return ENVIRONMENTS[env_name](config)


__all__ = [
    'BaseEnvironment',
    'CartPoleEnvironment',
    'create_environment',
    'ENVIRONMENTS'
]

