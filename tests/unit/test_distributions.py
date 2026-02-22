import torch
import numpy as np 
import pytest 



from rl_agents.distributions import TanhNormal, make_action_distribution





class TestTanhNormal:
    """Test suite for TanhNormal distribution"""
    
    def test_initialization(self):
        """Test TanhNormal can be initialized"""
        mu = torch.zeros(32, 6)
        std = torch.ones(32, 6)
        
        # Without bounds
        dist1 = TanhNormal(mu, std)
        assert dist1._scale == 1.0
        assert dist1._bias == 0.0
        
        # With bounds
        dist2 = TanhNormal(mu, std, low=-1.0, high=1.0)
        assert dist2._scale == 1.0
        assert dist2._bias == 0.0
        
        # Different bounds
        dist3 = TanhNormal(mu, std, low=-2.0, high=3.0)
        assert torch.isclose(dist3._scale, torch.tensor(2.5))
        assert torch.isclose(dist3._bias, torch.tensor(0.5))


        # Tensor bounds (same for all dimensions)
        low_tensor = torch.tensor([-1.0] * 6)
        high_tensor = torch.tensor([1.0] * 6)
        dist4 = TanhNormal(mu, std, low=low_tensor, high=high_tensor)
        assert torch.allclose(dist4._scale, torch.tensor(1.0))
        assert torch.allclose(dist4._bias, torch.tensor(0.0))


        # Tensor bounds (different per dimension)
        low_mixed = torch.tensor([-2.0, -1.0, 0.0, -3.0, -1.5, -0.5])
        high_mixed = torch.tensor([2.0, 1.0, 1.0, 3.0, 1.5, 0.5])
        dist5 = TanhNormal(mu, std, low=low_mixed, high=high_mixed)

        expected_scale = (high_mixed - low_mixed) / 2.0
        expected_bias = (high_mixed + low_mixed) / 2.0

        assert torch.allclose(dist5._scale, expected_scale), \
            f"Expected scale {expected_scale}, got {dist5._scale}"
        assert torch.allclose(dist5._bias, expected_bias), \
            f"Expected bias {expected_bias}, got {dist5._bias}"
        

        # Verify shapes match
        assert dist5._scale.shape == (6,), f"Scale shape mismatch: {dist5._scale.shape}"
        assert dist5._bias.shape == (6,), f"Bias shape mismatch: {dist5._bias.shape}"


        # Tensor bounds with broadcasting
        low_broadcast = torch.tensor([-1.0])  # Will broadcast to [6]
        high_broadcast = torch.tensor([1.0])
        dist6 = TanhNormal(mu, std, low=low_broadcast, high=high_broadcast)
        assert torch.allclose(dist6._scale, torch.tensor(1.0))
        assert torch.allclose(dist6._bias, torch.tensor(0.0))


    
    def test_sample_shapes(self):
        """Test sampling produces correct shapes"""
        batch_size, action_dim = 32, 6
        mu = torch.zeros(batch_size, action_dim)
        std = torch.ones(batch_size, action_dim)
        dist = TanhNormal(mu, std, low=-1.0, high=1.0)
        
        # Sample
        action, raw_u = dist.sample()
        assert action.shape == (batch_size, action_dim)
        assert raw_u.shape == (batch_size, action_dim)
        
        # Rsample (reparameterized)
        action2, raw_u2 = dist.rsample()
        assert action2.shape == (batch_size, action_dim)
        assert raw_u2.shape == (batch_size, action_dim)


    
    def test_action_bounds(self):
        """Test that sampled actions are within bounds"""

        # (batch_size, action_dim)
        mu = torch.zeros(1000, 3)
        std = torch.ones(1000, 3)
        
        # Test different bounds
        bounds = [
            (-1.0, 1.0),
            (-2.0, 2.0),
            (0.0, 1.0),
            (-5.0, 5.0),
            (torch.tensor([0.0, -1.0, -0.5]), torch.tensor([0.5, 1.0, -0.1]))
        ]
        
        for low, high in bounds:
            dist = TanhNormal(mu, std, low=low, high=high)
            action, _ = dist.sample()
            
            # Check bounds (with small tolerance for numerical errors)
            assert torch.all(action >= low - 1e-6), f"Actions below {low}: {action.min()}"
            assert torch.all(action <= high + 1e-6), f"Actions above {high}: {action.max()}"

        
        # (seq_len, batch_size, action_dim) for Recurrent Policy output
        mu = torch.zeros(32, 64, 3)
        std = torch.ones(32, 64, 3)

        # Test different bounds
        bounds = [
            (-1.0, 1.0),
            (-2.0, 2.0),
            (0.0, 1.0),
            (-5.0, 5.0),
            (torch.tensor([0.0, -1.0, -0.5]), torch.tensor([0.5, 1.0, -0.1]))
        ]
        
        for low, high in bounds:
            dist = TanhNormal(mu, std, low=low, high=high)
            action, _ = dist.sample()
            
            # Check bounds (with small tolerance for numerical errors)
            assert torch.all(action >= low - 1e-6), f"Actions below {low}: {action.min()}"
            assert torch.all(action <= high + 1e-6), f"Actions above {high}: {action.max()}"


    def test_log_prob_from_u_stability(self):
        """Test that log_prob_from_u is numerically stable"""
        mu = 10*torch.ones(100, 3)
        std = torch.ones(100, 3)
        dist = TanhNormal(mu, std, low=-1.0, high=1.0)
        
        # Sample and compute log prob
        action, raw_u = dist.rsample()
        log_prob = dist.log_prob_from_u(raw_u)
        
        # Check no NaN or Inf
        assert not torch.isnan(log_prob).any(), "NaN in log_prob"
        assert not torch.isinf(log_prob).any(), "Inf in log_prob"
        assert torch.all(torch.isfinite(log_prob)), "Log prob should be finite"


        mu = -10*torch.ones(100, 3)
        std = torch.ones(100, 3)
        dist = TanhNormal(mu, std, low=-1.0, high=1.0)
        
        # Sample and compute log prob
        action, raw_u = dist.rsample()
        log_prob = dist.log_prob_from_u(raw_u)
        
        # Check no NaN or Inf
        assert not torch.isnan(log_prob).any(), "NaN in log_prob"
        assert not torch.isinf(log_prob).any(), "Inf in log_prob"
        assert torch.all(torch.isfinite(log_prob)), "Log prob should be finite"

    def test_entropy(self):
        """Test entropy computation"""
        mu = torch.zeros(10, 3)
        std = torch.ones(10, 3)
        dist = TanhNormal(mu, std, low=-1.0, high=1.0)
        
        entropy = dist.entropy()
        
        # Entropy should be positive
        assert torch.all(entropy > 0), "Entropy should be positive"
        
        # Check shape
        assert entropy.shape == (10, 3), f"Wrong entropy shape: {entropy.shape}"
    


    







if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])