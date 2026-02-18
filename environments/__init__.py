from .base_env import BaseEnvironment
from .cartpole import CartPoleEnvironment


# Registry of available environments
ENVIRONMENTS = {
    'cartpole': CartPoleEnvironment,
    # other environments
}



def create_environment(env_name: str, config: dict) -> BaseEnvironment:
    """
    Factory function to create environments
    
    Args:
        env_name: Name of environment (e.g., 'cartpole')
        config: Configuration dictionary for the environment

    Returns:
        BaseEnvironment instance

    Raises:
        ValueError: If env_name not found in registry
    """
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available: {list(ENVIRONMENTS.keys())}"
        )
    
    return ENVIRONMENTS[env_name](config)





