import pytest
import numpy as np 


from environments import create_environment


class TestSingleEnvironment:
    """Tests for single environment (num_envs=1)"""
    
    def test_cartpole_standard(self):
        """Test standard CartPole (wind_mag=0)"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 1
        }
        
        env = create_environment("cartpole", config)
        
        # Check it's not vectorized
        assert not env.is_vectorized
        assert env.num_envs == 1
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        # Should behave like standard CartPole
        for _ in range(10):
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs.shape == (4,)
            assert np.isscalar(reward)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            
            if terminated or truncated:
                break
        
        env.close()
    
    def test_cartpole_with_rightwind(self):
        """Test CartPole with right wind"""
        config = {
            "wind_mag": 3.0,
            "force_mag": 10.0,
            "num_envs": 1
        }
        
        env = create_environment("cartpole", config)
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        # Run a few steps
        for _ in range(10):
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isscalar(reward)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            if terminated or truncated:
                break
        
        env.close()
    
    def test_cartpole_with_leftwind(self):
        """Test CartPole with left wind"""
        config = {
            "wind_mag": -3.0,
            "force_mag": 10.0,
            "num_envs": 1
        }
        
        env = create_environment("cartpole", config)
        
        obs, info = env.reset()
        assert obs.shape == (4,)
        
        # Run a few steps
        for _ in range(10):
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isscalar(reward)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            if terminated or truncated:
                break
        
        env.close()
    
    def test_reward_shaping(self):
        """Test that reward is based on pole angle (cosine shaping)"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 1
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
    
    def test_reset_with_seed(self):
        """Test deterministic reset with seed"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 1
        }
        
        env = create_environment("cartpole", config)
        
        # Reset with same seed should give same observation
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.allclose(obs1, obs2), "Same seed should give same initial state"
        
        # Different seed should give different observation
        obs3, _ = env.reset(seed=123)
        assert not np.allclose(obs1, obs3), "Different seed should give different state"
        
        env.close()


class TestVectorizedEnvironmentSync:
    """Tests for vectorized environment with sync mode"""
    
    def test_vectorized_creation(self):
        """Test creating vectorized environment"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        # Check it's vectorized
        assert env.is_vectorized
        assert env.num_envs == 4
        assert env.vectorization_mode == "sync"
        
        env.close()
    
    def test_vectorized_reset(self):
        """Test reset returns correct shapes"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 8,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        obs, infos = env.reset()
        
        # Check shapes
        assert obs.shape == (8, 4), f"Expected (8, 4), got {obs.shape}"
        assert isinstance(infos, dict), "Infos should be a dict for VectorEnv"
        
        env.close()
    
    def test_vectorized_step(self):
        """Test step returns correct shapes"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        obs, infos = env.reset()
        
        # Sample random actions for all environments
        actions = np.array([env.env.single_action_space.sample() for _ in range(4)])
        
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Check shapes
        assert next_obs.shape == (4, 4), f"Expected (4, 4), got {next_obs.shape}"
        assert rewards.shape == (4,), f"Expected (4,), got {rewards.shape}"
        assert terminated.shape == (4,), f"Expected (4,), got {terminated.shape}"
        assert truncated.shape == (4,), f"Expected (4,), got {truncated.shape}"
        
        # Check types
        assert isinstance(rewards, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        
        env.close()
    
    def test_vectorized_autoreset(self):
        """Test that vectorized env auto-resets on episode end"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        obs, infos = env.reset()
        
        # Run until at least one episode terminates
        max_steps = 1000
        episode_ended = False
        
        for _ in range(max_steps):
            actions = np.array([env.env.single_action_space.sample() for _ in range(4)])
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Check if any environment ended
            if terminated.any() or truncated.any():
                episode_ended = True
                # Observation should still be valid (auto-reset)
                assert next_obs.shape == (4, 4)
                break
            
            obs = next_obs
        
        assert episode_ended, "At least one episode should have ended"
        
        env.close()
    
    def test_vectorized_with_wind(self):
        """Test vectorized environment with wind"""
        config = {
            "wind_mag": 2.0,
            "force_mag": 10.0,
            "num_envs": 8,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        obs, infos = env.reset()
        
        # Run for a few steps
        for _ in range(10):
            actions = np.array([env.env.single_action_space.sample() for _ in range(8)])
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            assert obs.shape == (8, 4)
            assert rewards.shape == (8,)
        
        env.close()
    
    def test_vectorized_reset_with_seed(self):
        """Test deterministic reset with seed in vectorized env"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "sync"
        }
        
        env = create_environment("cartpole", config)
        
        # Reset with same seed should give same observations
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.allclose(obs1, obs2), "Same seed should give same initial states"
        
        # Different seed should give different observations
        obs3, _ = env.reset(seed=123)
        assert not np.allclose(obs1, obs3), "Different seed should give different states"
        
        env.close()


class TestVectorizedEnvironmentAsync:
    """Tests for vectorized environment with async mode"""
    
    def test_async_creation(self):
        """Test creating async vectorized environment"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "async"
        }
        
        env = create_environment("cartpole", config)
        
        # Check it's vectorized with async mode
        assert env.is_vectorized
        assert env.num_envs == 4
        assert env.vectorization_mode == "async"
        
        env.close()
    
    def test_async_reset_and_step(self):
        """Test async vectorized environment reset and step"""
        config = {
            "wind_mag": 0.0,
            "force_mag": 10.0,
            "num_envs": 4,
            "vectorization_mode": "async"
        }
        
        env = create_environment("cartpole", config)
        
        obs, infos = env.reset()
        assert obs.shape == (4, 4)
        
        # Step
        actions = np.array([env.env.single_action_space.sample() for _ in range(4)])
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        assert next_obs.shape == (4, 4)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        
        env.close()


class TestEnvironmentProperties:
    """Tests for environment properties"""
    
    def test_get_state_dim(self):
        """Test get_state_dim method"""
        config = {"num_envs": 1}
        env = create_environment("cartpole", config)
        assert env.get_state_dim() == 4
        env.close()
    
    def test_get_action_dim(self):
        """Test get_action_dim method"""
        config = {"num_envs": 1}
        env = create_environment("cartpole", config)
        assert env.get_action_dim() == 2
        env.close()
    
    def test_is_discrete(self):
        """Test is_discrete method"""
        config = {"num_envs": 1}
        env = create_environment("cartpole", config)
        assert env.is_discrete() is True
        env.close()
    
    def test_get_action_bounds(self):
        """Test get_action_bounds method"""
        config = {"num_envs": 1}
        env = create_environment("cartpole", config)
        assert env.get_action_bounds() is None  # Discrete action space
        env.close()
    
    def test_repr(self):
        """Test string representation"""
        # Single env
        config = {"num_envs": 1}
        env = create_environment("cartpole", config)
        repr_str = repr(env)
        assert "Single" in repr_str
        assert "num_envs=1" in repr_str
        env.close()
        
        # Vectorized sync
        config = {"num_envs": 4, "vectorization_mode": "sync"}
        env = create_environment("cartpole", config)
        repr_str = repr(env)
        assert "Vectorized-SYNC" in repr_str
        assert "num_envs=4" in repr_str
        env.close()
        
        # Vectorized async
        config = {"num_envs": 4, "vectorization_mode": "async"}
        env = create_environment("cartpole", config)
        repr_str = repr(env)
        assert "Vectorized-ASYNC" in repr_str
        env.close()


class TestEnvironmentFactory:
    """Tests for create_environment factory function"""
    
    def test_unknown_environment(self):
        """Test that unknown environment raises error"""
        with pytest.raises(ValueError, match="Unknown environment"):
            create_environment("unknown_env", {})
    
    def test_default_config(self):
        """Test environment with minimal config"""
        config = {}  # Should use defaults
        env = create_environment("cartpole", config)
        
        # Should default to single environment
        assert env.num_envs == 1
        assert not env.is_vectorized
        
        env.close()


if __name__ == "__main__":
    # Run all tests
    print("Testing Single Environment...")
    test_single = TestSingleEnvironment()
    test_single.test_cartpole_standard()
    test_single.test_cartpole_with_rightwind()
    test_single.test_cartpole_with_leftwind()
    test_single.test_reward_shaping()
    test_single.test_reset_with_seed()
    print("✓ Single environment tests passed!")
    
    print("\nTesting Vectorized Environment (Sync)...")
    test_vec_sync = TestVectorizedEnvironmentSync()
    test_vec_sync.test_vectorized_creation()
    test_vec_sync.test_vectorized_reset()
    test_vec_sync.test_vectorized_step()
    test_vec_sync.test_vectorized_autoreset()
    test_vec_sync.test_vectorized_with_wind()
    test_vec_sync.test_vectorized_reset_with_seed()
    print("✓ Vectorized sync tests passed!")
    
    print("\nTesting Vectorized Environment (Async)...")
    test_vec_async = TestVectorizedEnvironmentAsync()
    test_vec_async.test_async_creation()
    test_vec_async.test_async_reset_and_step()
    print("✓ Vectorized async tests passed!")
    
    print("\nTesting Environment Properties...")
    test_props = TestEnvironmentProperties()
    test_props.test_get_state_dim()
    test_props.test_get_action_dim()
    test_props.test_is_discrete()
    test_props.test_get_action_bounds()
    test_props.test_repr()
    print("✓ Property tests passed!")
    
    print("\nTesting Environment Factory...")
    test_factory = TestEnvironmentFactory()
    test_factory.test_default_config()
    print("✓ Factory tests passed!")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)