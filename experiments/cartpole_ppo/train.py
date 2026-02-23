"""
PPO Training Script for CartPole

This script trains a PPO agent on the CartPole environment with:
- Configurable hyperparameters via YAML
- Periodic evaluation
- Best model tracking and saving
- Checkpointing
- Logging and visualization
- Support for single and vectorized environments
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rl_agents import PPOAgent
from environments import create_environment
from utils import evaluate_agent, evaluate_agent_during_training, save_evaluation_results


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
    """
    Tracks best model based on evaluation metrics
    
    Attributes:
        best_reward: Best evaluation reward seen so far
        best_update: Update step when best model was found
        best_timestep: Timestep when best model was found
        checkpoint_path: Path where best model is saved
    """
    
    def __init__(self, checkpoint_dir: Path, metric: str = 'mean_reward'):
        """
        Initialize tracker
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric: Metric to track ('mean_reward', 'min_reward', 'max_reward')
        """
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.best_reward = -np.inf
        self.best_update = 0
        self.best_timestep = 0
        self.checkpoint_path = checkpoint_dir / 'best_model.pt'
        self.metadata_path = checkpoint_dir / 'best_model_metadata.json'
        
        # Create metadata file
        self._save_metadata()
    
    def update(self, agent, eval_reward: float, update: int, timestep: int) -> bool:
        """
        Check if current model is better and save if so
        
        Args:
            agent: Agent to save
            eval_reward: Current evaluation reward
            update: Current update number
            timestep: Current timestep
        
        Returns:
            True if new best model, False otherwise
        """
        if eval_reward > self.best_reward:
            self.best_reward = eval_reward
            self.best_update = update
            self.best_timestep = timestep
            
            # Save model
            agent.save(self.checkpoint_path)
            
            # Save metadata
            self._save_metadata()
            
            return True
        return False
    
    def _save_metadata(self):
        """Save metadata about best model"""
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
        """Get summary string"""
        if self.best_reward == -np.inf:
            return "No best model saved yet"
        return (f"Best model: reward={self.best_reward:.2f}, "
                f"update={self.best_update}, timestep={self.best_timestep}")


def train(config_path: str = None):
    """
    Main training function
    
    Args:
        config_path: Path to config file. If None, uses default config.
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    config = load_config(config_path)
    
    print("="*70)
    print("PPO TRAINING - CARTPOLE")
    print("="*70)
    print(f"Config loaded from: {config_path}")
    print(f"Device: {config['agent'].get('device', 'cpu')}")
    print(f"Total timesteps: {config['training']['total_timesteps']}")
    print(f"Num envs: {config['environment'].get('num_envs', 1)}")
    print("="*70 + "\n")
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(config['training'].get('checkpoint_dir', 'checkpoints'))
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")
    
    # Save config to checkpoint directory
    save_config(config, checkpoint_dir / 'config.yaml')
    
    # Create environments
    print("Creating environments...")
    
    # Training environment
    train_env = create_environment(
        config['environment']['name'],
        config['environment']
    )
    
    # Evaluation environment (single env for consistency)
    eval_config = config['environment'].copy()
    eval_config['num_envs'] = 1  # Always use single env for evaluation
    eval_env = create_environment(
        config['environment']['name'],
        eval_config
    )
    
    print(f"✓ Training environment created: {train_env}")
    print(f"✓ Evaluation environment created: {eval_env}\n")
    
    # Get environment info
    state_dim = train_env.get_state_dim()
    action_dim = train_env.get_action_dim()
    is_discrete = train_env.is_discrete()
    
    print(f"Environment info:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action space: {'Discrete' if is_discrete else 'Continuous'}\n")
    
    # Create agent
    print("Creating PPO agent...")
    agent_config = config['agent'].copy()
    agent_config['state_dim'] = state_dim
    agent_config['action_dim'] = action_dim
    agent_config['is_discrete'] = is_discrete
    
    agent = PPOAgent(agent_config)
    print(f"✓ Agent created")
    print(f"  Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"  Learning rate: {agent.learning_rate}")
    print(f"  LR scheduler: {agent_config.get('lr_scheduler', 'none')}\n")
    
    # Training parameters
    total_timesteps = config['training']['total_timesteps']
    n_steps = agent_config['n_steps']
    num_envs = config['environment'].get('num_envs', 1)
    n_updates = total_timesteps // (n_steps * num_envs)
    
    eval_frequency = config['training'].get('eval_frequency', 10)
    eval_episodes = config['training'].get('eval_episodes', 10)
    save_frequency = config['training'].get('save_frequency', 50)
    
    # Initialize best model tracker
    best_model_tracker = BestModelTracker(checkpoint_dir, metric='mean_reward')
    
    # Logging
    training_log = {
        'updates': [],
        'timesteps': [],
        'mean_rewards': [],
        'eval_rewards': [],
        'eval_timesteps': [],  # Track when evaluations happened
        'losses': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'approx_kls': [],
        'clip_fractions': [],
        'learning_rates': [],
        'best_reward_history': []  # Track best reward over time
    }
    
    # Training loop
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    is_vectorized = train_env.is_vectorized
    
    if is_vectorized:
        obs, _ = train_env.reset()
    else:
        obs, _ = train_env.reset()
    
    episode_rewards = []
    current_episode_rewards = np.zeros(num_envs)
    
    for update in range(n_updates):
        # Collect rollout
        for step in range(n_steps):
            # Select action
            action, info = agent.select_action(obs)
            
            # Step environment
            if is_vectorized:
                next_obs, rewards, terminated, truncated, env_info = train_env.step(action)
                dones = np.logical_or(terminated, truncated)
                
                # Add to buffer
                agent.buffer.add(
                    obs=obs,
                    action=action,
                    reward=rewards,
                    value=info['value'],
                    log_prob=info['log_prob'],
                    done=dones,
                    raw_action=info['raw_action']
                )
                
                # Track episode rewards
                current_episode_rewards += rewards
                for i in range(num_envs):
                    if dones[i]:
                        episode_rewards.append(current_episode_rewards[i])
                        current_episode_rewards[i] = 0
                
                obs = next_obs
            else:
                next_obs, reward, terminated, truncated, env_info = train_env.step(action[0])
                done = terminated or truncated
                
                # Add to buffer
                agent.buffer.add(
                    obs=obs,
                    action=np.array([action[0]]),
                    reward=reward,
                    value=info['value'][0],
                    log_prob=info['log_prob'][0],
                    done=done,
                    raw_action=info['raw_action']
                )
                
                # Track episode rewards
                current_episode_rewards[0] += reward
                if done:
                    episode_rewards.append(current_episode_rewards[0])
                    current_episode_rewards[0] = 0
                    obs, _ = train_env.reset()
                else:
                    obs = next_obs
        
        # Update agent
        if is_vectorized:
            # Get last values
            _, info = agent.select_action(obs)
            last_values = info['value']
    
            # Squeeze to remove extra dimension: (4, 1) -> (4,)
            if last_values.ndim > 1:
                last_values = last_values.squeeze()
    
            # Now apply the mask
            last_values = np.where(dones, 0.0, last_values)
    
        else:
            last_val = 0.0 if done else agent.select_action(obs)[1]['value'][0]
            last_values = last_val

        
        
        metrics = agent.update(last_values)
        
        # Log metrics
        current_timestep = (update + 1) * n_steps * num_envs
        training_log['updates'].append(update + 1)
        training_log['timesteps'].append(current_timestep)
        training_log['losses'].append(metrics['loss'])
        training_log['policy_losses'].append(metrics['policy_loss'])
        training_log['value_losses'].append(metrics['value_loss'])
        training_log['entropies'].append(metrics['entropy'])
        training_log['approx_kls'].append(metrics['approx_kl'])
        training_log['clip_fractions'].append(metrics['clip_fraction'])
        training_log['learning_rates'].append(metrics['learning_rate'])
        
        # Compute mean reward from recent episodes
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-20:])
            training_log['mean_rewards'].append(mean_reward)
        else:
            mean_reward = 0.0
            training_log['mean_rewards'].append(0.0)
        
        # Periodic evaluation
        eval_reward = None
        if (update + 1) % eval_frequency == 0:
            print(f"\n{'='*70}")
            print(f"EVALUATION AT UPDATE {update+1}")
            print(f"{'='*70}")
            
            eval_reward = evaluate_agent_during_training(
                agent=agent,
                env=eval_env,
                num_episodes=eval_episodes,
                max_steps_per_episode=500
            )
            
            training_log['eval_rewards'].append(eval_reward)
            training_log['eval_timesteps'].append(current_timestep)
            
            # Check if best model
            is_best = best_model_tracker.update(agent, eval_reward, update + 1, current_timestep)
            
            # Track best reward over time
            training_log['best_reward_history'].append(best_model_tracker.best_reward)
            
            if is_best:
                print(f"🎉 NEW BEST MODEL!")
                print(f"   Eval reward: {eval_reward:.2f}")
                print(f"   Previous best: {best_model_tracker.best_reward:.2f}")
                print(f"   Saved to: {best_model_tracker.checkpoint_path}")
            else:
                print(f"   Eval reward: {eval_reward:.2f}")
                print(f"   Best so far: {best_model_tracker.best_reward:.2f} "
                      f"(update {best_model_tracker.best_update})")
            
            print(f"{'='*70}\n")
        
        # Print progress
        if (update + 1) % eval_frequency == 0:
            print(f"Update {update+1}/{n_updates} | "
                  f"Timesteps: {current_timestep}/{total_timesteps} | "
                  f"Mean Reward: {mean_reward:.2f} | "
                  f"Eval Reward: {eval_reward:.2f if eval_reward else 'N/A'} | "
                  f"Best: {best_model_tracker.best_reward:.2f} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f} | "
                  f"KL: {metrics['approx_kl']:.4f} | "
                  f"LR: {metrics['learning_rate']:.6f}")
        
        # Save periodic checkpoint
        if (update + 1) % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{update+1}.pt'
            agent.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path.name}")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    final_results = evaluate_agent(
        agent=agent,
        env=eval_env,
        num_episodes=100,
        max_steps_per_episode=500,
        deterministic=True,
        verbose=True
    )
    
    # Check if final model is best
    final_is_best = best_model_tracker.update(
        agent, 
        final_results['mean_reward'], 
        n_updates, 
        total_timesteps
    )
    
    if final_is_best:
        print(f"\n🎉 Final model is the BEST model!")
    
    # Save final model
    agent.save(checkpoint_dir / 'final_model.pt')
    print(f"\nFinal model saved to: {checkpoint_dir / 'final_model.pt'}")
    
    # Save training log
    log_path = checkpoint_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to: {log_path}")
    
    # Save final evaluation results
    save_evaluation_results(
        results=final_results,
        save_path=checkpoint_dir / 'final_evaluation.json',
        agent_name='PPO',
        additional_info={
            'config': config,
            'total_timesteps': total_timesteps,
            'best_eval_reward': best_model_tracker.best_reward,
            'best_update': best_model_tracker.best_update,
            'best_timestep': best_model_tracker.best_timestep
        }
    )
    
    # Save best model evaluation (using best model)
    print(f"\n{'='*70}")
    print("EVALUATING BEST MODEL")
    print(f"{'='*70}")
    print(f"Loading best model from update {best_model_tracker.best_update}")
    print(f"Best eval reward during training: {best_model_tracker.best_reward:.2f}\n")
    
    agent.load(best_model_tracker.checkpoint_path)
    best_model_results = evaluate_agent(
        agent=agent,
        env=eval_env,
        num_episodes=100,
        max_steps_per_episode=500,
        deterministic=True,
        verbose=True
    )
    
    save_evaluation_results(
        results=best_model_results,
        save_path=checkpoint_dir / 'best_model_evaluation.json',
        agent_name='PPO-Best',
        additional_info={
            'config': config,
            'best_update': best_model_tracker.best_update,
            'best_timestep': best_model_tracker.best_timestep,
            'training_eval_reward': best_model_tracker.best_reward
        }
    )
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total updates: {n_updates}")
    print(f"\nBest Model:")
    print(f"  Update: {best_model_tracker.best_update}")
    print(f"  Timestep: {best_model_tracker.best_timestep}")
    print(f"  Eval reward (during training): {best_model_tracker.best_reward:.2f}")
    print(f"  Eval reward (final check): {best_model_results['mean_reward']:.2f} ± {best_model_results['std_reward']:.2f}")
    print(f"\nFinal Model:")
    print(f"  Eval reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    return training_log, final_results, best_model_results, checkpoint_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent on CartPole')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config.yaml in same directory)'
    )
    
    args = parser.parse_args()
    
    # Run training
    train(config_path=args.config)