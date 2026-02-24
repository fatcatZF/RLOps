import yaml
import json 

from datetime import datetime
from pathlib import Path
import numpy as np 


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert scientific notation to numbers
    config['training']['total_timesteps'] = int(float(config['training']['total_timesteps']))
    config['agent']['total_timesteps'] = int(float(config['agent']['total_timesteps']))
    
    return config


def save_config(config: dict, save_path: str):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_checkpoint_dir(base_dir: str = 'checkpoints') -> Path:
    """Create timestamped checkpoint directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path(base_dir) / f'ppo_cartpole_{timestamp}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


class BestModelTracker:
    """Tracks best model based on evaluation metrics"""
    
    def __init__(self, checkpoint_dir: Path, metric: str = 'mean_reward'):
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.best_reward = -np.inf
        self.best_update = 0
        self.best_timestep = 0
        self.checkpoint_path = checkpoint_dir / 'best_model.pt'
        self.metadata_path = checkpoint_dir / 'best_model_metadata.json'
        self._save_metadata()
    
    def update(self, agent, eval_reward: float, update: int, timestep: int) -> bool:
        if eval_reward > self.best_reward:
            self.best_reward = eval_reward
            self.best_update = update
            self.best_timestep = timestep
            agent.save(self.checkpoint_path)
            self._save_metadata()
            return True
        return False
    
    def _save_metadata(self):
        metadata = {
            'best_reward': float(self.best_reward) if self.best_reward != -np.inf else None,
            'best_update': int(self.best_update),
            'best_timestep': int(self.best_timestep),
            'metric': self.metric,
            'checkpoint_path': str(self.checkpoint_path)
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_summary(self) -> str:
        if self.best_reward == -np.inf:
            return "No best model saved yet"
        return (f"Best model: reward={self.best_reward:.2f}, "
                f"update={self.best_update}, timestep={self.best_timestep}")