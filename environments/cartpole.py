import gymnasium as gym 
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, Wrapper

import numpy as np 

from typing import Dict, Any, Optional, Tuple

from .base_env import BaseEnvironment








class CartPoleEnvWithWind(CartPoleEnv):
    """CartPole environment with added wind force"""
    
    def __init__(self, force_mag=10.0, wind_mag=0.0, **kwargs):
        """
        Initialize CartPole with wind
        
        Args:
            force_mag: Magnitude of applied force
            wind_mag: Wind magnitude (should be in [-5, 5])
            **kwargs: Additional arguments passed to CartPoleEnv
        """
        super().__init__(**kwargs)
        self.force_mag = force_mag
        assert -5 <= wind_mag <= 5, "Wind magnitude should be within [-5, 5]"
        self.wind_mag = wind_mag

    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Apply wind force
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta + self.wind_mag
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else np.cos(theta)
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = -1.0 if self._sutton_barto_reward else np.cos(theta)
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already "
                    "returned terminated = True. You should always call 'reset()' once "
                    "you receive 'terminated = True'"
                )
            self.steps_beyond_terminated += 1
            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    



# Register environment
gym.register(
    id='CartPoleWithWind-v1',
    entry_point='environments.cartpole:CartPoleEnvWithWind',
    max_episode_steps=500
)



class CartPoleEnvironment(BaseEnvironment):
    """
    CartPole environment wrapper for RLOps framework
    Supports standard CartPole and CartPole with wind
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CartPole environment
        
        Args:
            config: Configuration dictionary with keys:
                - variant: 'standard' or 'wind' (default: 'standard')
                - wind_mag: Wind magnitude if variant='wind' (default: 0.0)
                - force_mag: Force magnitude (default: 10.0)
                - reward_shaping: Whether to use reward shaping (default: False)
                - max_episode_steps: Maximum steps per episode (default: 500)
        """
        super().__init__(config)
        self.wind_mag = config.get('wind_mag', 0.0)
        self.force_mag = config.get('force_mag', 10.0)
        
        self.env = self.make_env()
    
    def make_env(self) -> gym.Env:
        """Create the CartPole environment"""
        env = CartPoleEnvWithWind(
                force_mag=self.force_mag,
                wind_mag=self.wind_mag
        )
        
        return env
    
    def get_state_dim(self) -> int:
        """CartPole has 4-dimensional state space"""
        return 4
    
    def get_action_dim(self) -> int:
        """CartPole has 2 discrete actions"""
        return 2
    
    def is_discrete(self) -> bool:
        """CartPole has discrete action space"""
        return True
    
    def get_action_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """CartPole has discrete actions, so no bounds"""
        return None

