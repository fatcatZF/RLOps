import torch
import torch.nn as nn
from typing import Tuple, Optional, Type


class MLP(nn.Module):
    """
    Multi-layer perceptron with customizable activations
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Tuple of hidden layer dimensions
        activation: Activation function class (default: nn.ReLU)
        output_activation: Optional activation for output layer (default: None)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Optional output activation
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    


