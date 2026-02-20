import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Type, Any 


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
    








class ActorCriticWithSharedFeature(nn.Module):
    """
    Actor-Critic with shared feature extractor
    
    The feature extractor can be:
    - MLP for vector observations
    - CNN for image observations  
    - LSTM/GRU for sequential observations
    - Any custom architecture
    
    Args:
        feature_extractor: Neural network that extracts features.
                          Must have .output_dim attribute.
        action_dim: Action space dimension
        is_discrete: Whether action space is discrete
    
    Example:
        >>> # With MLP feature extractor
        >>> feature_extractor = MLP(input_dim=4, output_dim=64)
        >>> network = ActorCriticShared(feature_extractor, action_dim=2)
        >>> 
        >>> # With CNN feature extractor
        >>> feature_extractor = CNN(input_channels=4, output_dim=512)
        >>> network = ActorCriticShared(feature_extractor, action_dim=6)
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        action_dim: int,
        is_discrete: bool = True
    ):
        super().__init__()
        
        # Validate feature extractor
        if not hasattr(feature_extractor, 'output_dim'):
            raise ValueError(
                "Feature extractor must have 'output_dim' attribute. "
                "Add self.output_dim = <dim> in your feature extractor."
            )
        
        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.feature_dim = feature_extractor.output_dim
        
        # Actor head
        if is_discrete:
            self.actor_head = nn.Linear(self.feature_dim, action_dim)
        else:
            self.actor_head = nn.Linear(self.feature_dim, 2 * action_dim)
        
        # Critic head
        self.critic_head = nn.Linear(self.feature_dim, 1)
    
    def forward(
        self, 
        observation: torch.Tensor
    ) -> Tuple[ActorOutput, torch.Tensor]:
        """Forward pass through shared features and heads"""
        features = self.feature_extractor(observation)
        value = self.critic_head(features)
        
        if self.is_discrete:
            logits = self.actor_head(features)
            return logits, value
        else:
            actor_output = self.actor_head(features)
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            log_std = torch.clamp(log_std, min=-20, max=2)
            return (mean, log_std), value
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate"""
        features = self.feature_extractor(observation)
        return self.critic_head(features)
    
    def get_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Get shared features"""
        return self.feature_extractor(observation)
    






class ActorCritic(nn.Module):
    """
    Actor-Critic with separate networks (no feature sharing)
    
    Each network (actor and critic) has its own independent architecture.
    
    Args:
        state_dim: Observation space dimension
        action_dim: Action space dimension
        hidden_dims: Hidden layer dimensions for both networks
        activation: Activation function
        is_discrete: Whether action space is discrete
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: Union[str, Type[nn.Module]] = 'relu',
        is_discrete: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        # Separate actor network
        if is_discrete:
            self.actor = MLP(state_dim, action_dim, hidden_dims, activation)
        else:
            self.actor = MLP(state_dim, 2 * action_dim, hidden_dims, activation)
        
        # Separate critic network
        self.critic = MLP(state_dim, 1, hidden_dims, activation)
    
    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[ActorOutput, torch.Tensor]:
        """Forward pass through separate networks"""
        value = self.critic(state)
        
        if self.is_discrete:
            logits = self.actor(state)
            return logits, value
        else:
            actor_output = self.actor(state)
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            log_std = torch.clamp(log_std, min=-20, max=2)
            return (mean, log_std), value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate"""
        return self.critic(state)








def create_actor_critic(
    state_dim: Optional[int] = None,
    action_dim: int = 2,
    is_discrete: bool = True,
    observation_type: str = 'vector',
    observation_shape: Optional[Tuple[int, ...]] = None,
    shared_features: bool = True,
    feature_dim: int = 64,
    hidden_dims: Tuple[int, ...] = (64, 64),
    activation: Union[str, Type[nn.Module]] = 'relu',
    **kwargs
) -> nn.Module:
    """
    Factory function to create Actor-Critic networks
    
    Args:
        state_dim: State dimension (for vector observations)
        action_dim: Action dimension
        is_discrete: Whether action space is discrete
        observation_type: 'vector' or 'image'
        observation_shape: Shape of observation (e.g., (4,) or (4, 84, 84))
        shared_features: If True, use shared feature extractor
        feature_dim: Dimension of extracted features
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        **kwargs: Additional arguments for specific architectures
    
    Returns:
        Actor-Critic network (ActorCriticWithSharedFeature or ActorCritic)
    
    Examples:
        >>> # CartPole (vector, shared features)
        >>> network = create_actor_critic(
        ...     state_dim=4,
        ...     action_dim=2,
        ...     is_discrete=True,
        ...     observation_type='vector',
        ...     shared_features=True
        ... )
        
        >>> # Atari (image, shared features)
        >>> network = create_actor_critic(
        ...     observation_type='image',
        ...     observation_shape=(4, 84, 84),
        ...     action_dim=6,
        ...     is_discrete=True,
        ...     shared_features=True,
        ...     feature_dim=512
        ... )
        
        >>> # MuJoCo (vector, separate networks)
        >>> network = create_actor_critic(
        ...     state_dim=17,
        ...     action_dim=6,
        ...     is_discrete=False,
        ...     shared_features=False
        ... )
    """
    # Determine observation shape
    if observation_shape is None:
        if state_dim is None:
            raise ValueError("Must provide either state_dim or observation_shape")
        observation_shape = (state_dim,)
    
    # Create separate networks (no feature sharing)
    if not shared_features:
        if observation_type != 'vector':
            raise ValueError(
                "Separate networks only supported for vector observations. "
                "Use shared_features=True for image observations."
            )
        
        return ActorCritic(
            state_dim=observation_shape[0],
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            is_discrete=is_discrete
        )
    
    # Create shared feature extractor
    if observation_type == 'vector':
        feature_extractor = MLP(
            input_dim=observation_shape[0],
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    elif observation_type == 'image':
        if len(observation_shape) != 3:
            raise ValueError(
                f"Image observations must have shape (channels, height, width). "
                f"Got: {observation_shape}"
            )
        
        channels, height, width = observation_shape
        feature_extractor = CNN(
            input_channels=channels,
            output_dim=feature_dim,
            height=height,
            width=width
        )
    
    else:
        raise ValueError(
            f"Unknown observation_type: '{observation_type}'. "
            f"Supported: 'vector', 'image'"
        )
    
    # Create Actor-Critic with shared features
    return ActorCriticWithSharedFeature(
        feature_extractor=feature_extractor,
        action_dim=action_dim,
        is_discrete=is_discrete
    )







def create_actor_critic_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create Actor-Critic from configuration dictionary
    
    Args:
        config: Configuration dict with keys:
            - network_type: 'shared' or 'separate'
            - observation_type: 'vector' or 'image'
            - state_dim or observation_shape
            - action_dim
            - is_discrete
            - feature_dim (for shared)
            - hidden_dims
            - activation
    
    Returns:
        Actor-Critic network
    
    Example:
        >>> config = {
        ...     'network_type': 'shared',
        ...     'observation_type': 'vector',
        ...     'state_dim': 4,
        ...     'action_dim': 2,
        ...     'is_discrete': True,
        ...     'feature_dim': 64,
        ...     'hidden_dims': [64, 64],
        ...     'activation': 'tanh'
        ... }
        >>> network = create_actor_critic_from_config(config)
    """
    network_type = config.get('network_type', 'shared')
    observation_type = config.get('observation_type', 'vector')
    
    # Get observation shape
    if 'observation_shape' in config:
        observation_shape = tuple(config['observation_shape'])
    elif 'state_dim' in config:
        observation_shape = (config['state_dim'],)
    else:
        raise ValueError("Must provide 'observation_shape' or 'state_dim' in config")
    
    action_dim = config['action_dim']
    is_discrete = config.get('is_discrete', True)
    
    # Separate networks
    if network_type == 'separate':
        if observation_type != 'vector':
            raise ValueError("Separate networks only for vector observations")
        
        return ActorCritic(
            state_dim=observation_shape[0],
            action_dim=action_dim,
            hidden_dims=tuple(config.get('hidden_dims', [64, 64])),
            activation=config.get('activation', 'relu'),
            is_discrete=is_discrete
        )
    
    # Shared features
    elif network_type == 'shared':
        feature_dim = config.get('feature_dim', 64)
        hidden_dims = tuple(config.get('hidden_dims', [64, 64]))
        activation = config.get('activation', 'relu')
        
        # Create feature extractor
        if observation_type == 'vector':
            feature_extractor = MLP(
                input_dim=observation_shape[0],
                output_dim=feature_dim,
                hidden_dims=hidden_dims,
                activation=activation
            )
        elif observation_type == 'image':
            channels, height, width = observation_shape
            feature_extractor = CNN(
                input_channels=channels,
                output_dim=feature_dim,
                height=height,
                width=width
            )
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}")
        
        return ActorCriticWithSharedFeature(
            feature_extractor=feature_extractor,
            action_dim=action_dim,
            is_discrete=is_discrete
        )
    
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
