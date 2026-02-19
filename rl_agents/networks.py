import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Type


# Type Alias
ActorOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] # logits for discrete action or (mu, logstd) for continuous action



class MLP(nn.Module):
    """
    Multi-layer perceptron with customizable activations
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Tuple of hidden layer dimensions
        activation: Activation function class (default: nn.ReLU)
        output_activation: Optional activation for output layer (default: None)
    Example:
        >>> # Using class
        >>> mlp = MLP(4, 2, (64, 64), activation=nn.Tanh)
        >>> 
        >>> # Using string (config-friendly)
        >>> mlp = MLP(4, 2, (64, 64), activation='tanh')
    """

    # Mapping of activation names to classes
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
        'gelu': nn.GELU,
        'selu': nn.SELU,
        'softplus': nn.Softplus,
    }
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: Union[str, Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Union[str, Type[nn.Module]]] = None
    ):
        super().__init__()

        # Convert string to activation class if needed
        if isinstance(activation, str):
            if activation.lower() not in self.ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation: '{activation}'. "
                    f"Available: {list(self.ACTIVATIONS.keys())}"
                )
            activation = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)
        
        if isinstance(output_activation, str):
            if output_activation.lower() not in self.ACTIVATIONS:
                raise ValueError(
                    f"Unknown output activation: '{output_activation}'. "
                    f"Available: {list(self.ACTIVATIONS.keys())}"
                )
            output_activation = self.ACTIVATIONS.get(output_activation.lower())

        
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
    




class CNN(nn.Module):
    """
    Convolutional Neural Network for image inputs
    
    Typical for visual observations
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        height: int = 84,
        width: int = 84
    ):
        super().__init__()
        
        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size (depends on input image size)
        # For 84x84 Atari images, this gives 7*7*64 = 3136
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, height, width)
            conv_output_size = self.conv_net(sample_input).shape[1]
        
        # Linear layer to desired output dimension
        self.linear = nn.Linear(conv_output_size, output_dim)
        
        # Store output dimension
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, height, width]
        
        Returns:
            features: [batch_size, output_dim]
        """
        conv_features = self.conv_net(x)
        return self.linear(conv_features)
    








