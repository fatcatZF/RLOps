"""
Base trainer class for RL algorithms

Provides common training functionality that can be extended by algorithm-specific trainers.
"""

import numpy as np
from typing import Dict, Any
from datetime import datetime
import json
from abc import ABC, abstractmethod

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.experiments import (
    create_checkpoint_dir,
    save_config,
    BestModelTracker
)

from utils.evaluation import (
    evaluate_agent,
    evaluate_agent_during_training,
    save_evaluation_results
)


class BaseTrainer(ABC):
    """
    Base trainer class for RL algorithms
    
    Provides common training functionality including:
    - Training loop structure
    - Evaluation
    - Checkpointing
    - Logging (W&B)
    - Progress tracking
    
    Subclasses should implement algorithm-specific methods:
    - _collect_rollout()
    - _compute_last_values()
    
    Args:
        agent: RL agent instance
        train_env: Training environment
        eval_env: Evaluation environment (optional)
        config: Training configuration dictionary
        experiment_name: Name for this experiment
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        verbose: Verbosity level (0: quiet, 1: progress, 2: detailed)
    
    Example:
        >>> # Use via subclass (e.g., PPOTrainer)
        >>> from trainers import PPOTrainer
        >>> trainer = PPOTrainer(agent, train_env, eval_env, config)
        >>> results = trainer.train()
    """
    
    def __init__(
        self,
        agent,
        train_env,
        eval_env=None,
        config: Dict[str, Any] = None,
        experiment_name: str = None,
        use_wandb: bool = True,
        wandb_project: str = 'rl-training',
        verbose: int = 1
    ):
        """Initialize base trainer"""
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env if eval_env is not None else train_env
        self.config = config or {}
        self.verbose = verbose
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'experiment_{timestamp}'
        self.experiment_name = experiment_name
        
        # Extract training parameters
        training_config = self.config.get('training', {})
        self.total_timesteps = training_config.get('total_timesteps', 1000000)
        self.eval_frequency = training_config.get('eval_frequency', 10)
        self.eval_episodes = training_config.get('eval_episodes', 10)
        self.save_frequency = training_config.get('save_frequency', 50)
        self.max_episode_steps = training_config.get('max_episode_steps', 1000)
        
        # Create checkpoint directory
        checkpoint_base = training_config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_base, prefix=experiment_name)
        
        # Save config
        save_config(self.config, self.checkpoint_dir / 'config.yaml')
        
        # Initialize best model tracker
        self.best_model_tracker = BestModelTracker(
            self.checkpoint_dir,
            metric='eval_reward'
        )
        
        # Initialize logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        
        if self.use_wandb:
            self._init_wandb(wandb_project)
        
        # Training state
        self.current_update = 0
        self.current_timestep = 0
        
        # Metrics log
        self.training_log = {
            'updates': [],
            'timesteps': [],
            'mean_rewards': [],
            'eval_rewards': [],
            'eval_timesteps': [],
        }
        
        # Environment info
        self.is_vectorized = hasattr(train_env, 'is_vectorized') and train_env.is_vectorized
        self.num_envs = train_env.num_envs if self.is_vectorized else 1
        
        if self.verbose > 0:
            self._print_init_info()
    
    def _init_wandb(self, project: str):
        """Initialize Weights & Biases"""
        try:
            self.wandb_run = wandb.init(
                project=project,
                config=self.config,
                name=self.experiment_name,
                dir=str(self.checkpoint_dir),
                save_code=True,
                reinit=True
            )
            
            # Watch model if available
            if hasattr(self.agent, 'network'):
                wandb.watch(self.agent.network, log='all', log_freq=100)
            
            if self.verbose > 0:
                print(f"✓ W&B run initialized: {self.wandb_run.name}")
                print(f"  Dashboard: {self.wandb_run.url}\n")
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            self.use_wandb = False
    
    def _print_init_info(self):
        """Print initialization information"""
        print("="*70)
        print(f"TRAINING: {self.experiment_name}")
        print("="*70)
        print(f"Algorithm: {self.__class__.__name__}")
        print(f"Training environment: {self.train_env}")
        print(f"Evaluation environment: {self.eval_env}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"W&B logging: {'Enabled' if self.use_wandb else 'Disabled'}")
        print("="*70 + "\n")
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop
        
        Returns:
            Dictionary with training results
        """
        # Get training parameters
        n_steps = self._get_n_steps()
        n_updates = self.total_timesteps // (n_steps * self.num_envs)
        
        if self.verbose > 0:
            self._print_training_start(n_updates, n_steps)
        
        # Initialize training
        obs, episode_rewards = self._initialize_training()
        
        # Training loop
        for update in range(n_updates):
            self.current_update = update + 1
            
            # Collect rollout (algorithm-specific)
            obs, episode_rewards = self._collect_rollout(obs, episode_rewards, n_steps)
            
            # Update agent (algorithm-specific)
            metrics = self._update_agent(obs)
            
            # Update timestep counter
            self.current_timestep = (update + 1) * n_steps * self.num_envs
            
            # Process metrics
            self._process_update_metrics(metrics, episode_rewards)
            
            # Periodic evaluation
            if (update + 1) % self.eval_frequency == 0:
                self._run_evaluation()
            
            # Print progress
            if (update + 1) % self.eval_frequency == 0 and self.verbose > 0:
                self._print_progress(n_updates, metrics)
            
            # Save periodic checkpoint
            if (update + 1) % self.save_frequency == 0:
                self._save_checkpoint()
        
        # Final evaluation and cleanup
        final_results = self._final_evaluation()
        self._finish_training(final_results)
        
        return {
            'best_reward': self.best_model_tracker.best_reward,
            'final_reward': final_results['mean_reward'],
            'checkpoint_dir': str(self.checkpoint_dir),
            'training_log': self.training_log
        }
    
    @abstractmethod
    def _collect_rollout(self, obs, episode_rewards, n_steps):
        """
        Collect rollout (algorithm-specific)
        
        Args:
            obs: Current observation(s)
            episode_rewards: List of episode rewards
            n_steps: Number of steps to collect
        
        Returns:
            Tuple of (obs, episode_rewards) after rollout
        """
        pass
    
    @abstractmethod
    def _update_agent(self, obs):
        """
        Update agent (algorithm-specific)
        
        Args:
            obs: Current observation(s) for computing last values
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def _get_n_steps(self) -> int:
        """Get number of steps per update"""
        return getattr(self.agent, 'n_steps', 2048)
    
    def _initialize_training(self):
        """Initialize training state"""
        obs, _ = self.train_env.reset()
        episode_rewards = []
        return obs, episode_rewards
    
    def _print_training_start(self, n_updates: int, n_steps: int):
        """Print training start information"""
        print("="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Total updates: {n_updates}")
        print(f"Steps per update: {n_steps}")
        print(f"Environments: {self.num_envs}")
        print(f"Total steps per update: {n_steps * self.num_envs}")
        print(f"Eval frequency: every {self.eval_frequency} updates")
        print(f"Save frequency: every {self.save_frequency} updates")
        print("="*70 + "\n")
    
    def _process_update_metrics(self, metrics: Dict[str, float], episode_rewards: list):
        """Process and log metrics from agent update"""
        # Log to internal tracking
        self.training_log['updates'].append(self.current_update)
        self.training_log['timesteps'].append(self.current_timestep)
        
        # Compute mean reward
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-20:])
            self.training_log['mean_rewards'].append(mean_reward)
        else:
            mean_reward = 0.0
            self.training_log['mean_rewards'].append(0.0)
        
        # Add agent metrics to log
        for key, value in metrics.items():
            if key not in self.training_log:
                self.training_log[key] = []
            self.training_log[key].append(value)
        
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {
                'train/mean_reward': mean_reward,
                'update': self.current_update,
                'timestep': self.current_timestep
            }
            for key, value in metrics.items():
                wandb_metrics[f'train/{key}'] = value
            wandb.log(wandb_metrics)
    
    def _run_evaluation(self):
        """Run evaluation"""
        if self.verbose > 1:
            print(f"\n{'='*70}")
            print(f"EVALUATION AT UPDATE {self.current_update}")
            print(f"{'='*70}")
        
        eval_reward = evaluate_agent_during_training(
            agent=self.agent,
            env=self.eval_env,
            num_episodes=self.eval_episodes,
            max_steps_per_episode=self.max_episode_steps
        )
        
        self.training_log['eval_rewards'].append(eval_reward)
        self.training_log['eval_timesteps'].append(self.current_timestep)
        
        # Check if best model
        is_best = self.best_model_tracker.update(
            self.agent,
            eval_reward,
            self.current_update,
            self.current_timestep
        )
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'eval/reward': eval_reward,
                'eval/best_reward': self.best_model_tracker.best_reward,
                'update': self.current_update,
                'timestep': self.current_timestep
            })
        
        if is_best:
            if self.verbose > 0:
                print(f"🎉 NEW BEST MODEL!")
                print(f"   Eval reward: {eval_reward:.2f}")
                print(f"   Saved to: {self.best_model_tracker.checkpoint_path}")
            
            if self.use_wandb:
                wandb.run.summary['best_eval_reward'] = eval_reward
                wandb.run.summary['best_update'] = self.current_update
                wandb.save(str(self.best_model_tracker.checkpoint_path))
        elif self.verbose > 1:
            print(f"   Eval reward: {eval_reward:.2f}")
            print(f"   Best so far: {self.best_model_tracker.best_reward:.2f}")
        
        if self.verbose > 1:
            print(f"{'='*70}\n")
    
    def _print_progress(self, n_updates: int, metrics: Dict[str, float]):
        """Print training progress"""
        eval_reward = self.training_log['eval_rewards'][-1] if self.training_log['eval_rewards'] else None
        mean_reward = self.training_log['mean_rewards'][-1]
        eval_str = f"{eval_reward:.2f}" if eval_reward is not None else "N/A"
        
        progress = (
            f"Update {self.current_update}/{n_updates} | "
            f"Timesteps: {self.current_timestep}/{self.total_timesteps} | "
            f"Mean Reward: {mean_reward:.2f} | "
            f"Eval Reward: {eval_str} | "
            f"Best: {self.best_model_tracker.best_reward:.2f}"
        )
        
        # Add algorithm-specific metrics
        progress = self._add_metrics_to_progress(progress, metrics)
        print(progress)
    
    def _add_metrics_to_progress(self, progress: str, metrics: Dict[str, float]) -> str:
        """Add algorithm-specific metrics to progress string (can be overridden)"""
        if 'loss' in metrics:
            progress += f" | Loss: {metrics['loss']:.4f}"
        return progress
    
    def _save_checkpoint(self):
        """Save periodic checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{self.current_update}.pt'
        self.agent.save(checkpoint_path)
        
        if self.verbose > 1:
            print(f"  → Checkpoint saved: {checkpoint_path.name}")
        
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Run final evaluation"""
        if self.verbose > 0:
            print("\n" + "="*70)
            print("FINAL EVALUATION")
            print("="*70)
        
        final_results = evaluate_agent(
            agent=self.agent,
            env=self.eval_env,
            num_episodes=100,
            max_steps_per_episode=self.max_episode_steps,
            deterministic=True,
            verbose=(self.verbose > 0)
        )
        
        # Check if final model is best
        self.best_model_tracker.update(
            self.agent,
            final_results['mean_reward'],
            self.current_update,
            self.current_timestep
        )
        
        # Save final model
        self.agent.save(self.checkpoint_dir / 'final_model.pt')
        
        if self.verbose > 0:
            print(f"Final model saved to: {self.checkpoint_dir / 'final_model.pt'}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'final/mean_reward': final_results['mean_reward'],
                'final/std_reward': final_results['std_reward'],
                'final/min_reward': final_results['min_reward'],
                'final/max_reward': final_results['max_reward']
            })
            wandb.save(str(self.checkpoint_dir / 'final_model.pt'))
        
        # Save evaluation results
        save_evaluation_results(
            results=final_results,
            save_path=self.checkpoint_dir / 'final_evaluation.json',
            agent_name=self.experiment_name,
            additional_info={
                'config': self.config,
                'total_timesteps': self.total_timesteps,
                'best_eval_reward': self.best_model_tracker.best_reward,
                'best_update': self.best_model_tracker.best_update,
                'best_timestep': self.best_model_tracker.best_timestep
            }
        )
        
        return final_results
    
    def _finish_training(self, final_results: Dict[str, Any]):
        """Cleanup and print summary"""
        # Save training log
        log_path = self.checkpoint_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        if self.verbose > 0:
            print(f"Training log saved to: {log_path}")
        
        # Print summary
        if self.verbose > 0:
            print("\n" + "="*70)
            print("TRAINING SUMMARY")
            print("="*70)
            print(f"Total timesteps: {self.total_timesteps:,}")
            print(f"Total updates: {self.current_update}")
            print(f"\nBest Model:")
            print(f"  Update: {self.best_model_tracker.best_update}")
            print(f"  Timestep: {self.best_model_tracker.best_timestep}")
            print(f"  Eval reward: {self.best_model_tracker.best_reward:.2f}")
            print(f"\nFinal Model:")
            print(f"  Eval reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
            print(f"\nCheckpoint directory: {self.checkpoint_dir}")
            if self.use_wandb:
                print(f"W&B dashboard: {self.wandb_run.url}")
            print("="*70 + "\n")
        
        # Close environments
        self.train_env.close()
        if self.eval_env != self.train_env:
            self.eval_env.close()
        
        # Finish W&B
        if self.use_wandb:
            wandb.finish()
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"{self.__class__.__name__}(experiment='{self.experiment_name}', "
                f"timesteps={self.total_timesteps:,}, updates={self.current_update})")







