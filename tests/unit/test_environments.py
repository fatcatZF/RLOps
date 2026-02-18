import pytest
import numpy as np 
from environments import create_environment


def test_cartpole_standard():
    """Test standard CartPole (wind_mag=0)"""
    config = {
        "wind_mag": 0.0,
        "force_mag": 10.0,
    }
    
    env = create_environment("cartpole", config)
    
    obs, info = env.reset()
    assert obs.shape == (4,)
    
    # Should behave like standard CartPole
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            env.env.action_space.sample()
        )
        if terminated or truncated:
            break
    
    env.close()






def test_cartpole_with_rightwind():
    """Test CartPole with wind"""
    config = {
        "wind_mag": 3.0,
        "force_mag": 10.0,
    }

    env = create_environment("cartpole", config)

    obs, info = env.reset()
    assert obs.shape == (4,)

    # Run a few steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            env.env.action_space.sample()
        )
        assert np.isscalar(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break
    
    env.close()




def test_cartpole_with_leftwind():
    """Test CartPole with wind"""
    config = {
        "wind_mag": -3.0,
        "force_mag": 10.0,
    }
    
    env = create_environment("cartpole", config)
    
    obs, info = env.reset()
    assert obs.shape == (4,)
    
    # Run a few steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            env.env.action_space.sample()
        )
        assert np.isscalar(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        if terminated or truncated:
            break
    
    env.close()


def test_reward_shaping():
    """Test that reward is based on pole angle (cosine shaping)"""
    config = {
        "wind_mag": 0.0,
        "force_mag": 10.0,
    }
    
    env = create_environment("cartpole", config)
    
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Reward should be cosine of pole angle (obs[2])
    pole_angle = obs[2]
    expected_reward = np.cos(pole_angle)
    assert np.isclose(reward, expected_reward), \
        f"Expected {expected_reward}, got {reward}"
    
    env.close()




if __name__ == "__main__":
    test_cartpole_standard()
    test_cartpole_with_rightwind()
    test_cartpole_with_leftwind()
    test_reward_shaping()

    print("All environment tests passed!")