"""
PPO Training Script for CartPole

This script trains a PPO agent on the CartPole environment with:
- Configurable hyperparameters via YAML
- Periodic evaluation
- Best model tracking and saving
- Checkpointing
- Logging and visualization
- Support for single and vectorized environments
- wandb integration
"""


import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import json
import wandb  

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_agents import PPOAgent
from environments import create_environment
from utils.evaluation import (
    evaluate_agent,
    evaluate_agent_during_training,
    save_evaluation_results
)

from utils.experiments import (
    load_config,
    save_config,
    create_checkpoint_dir,
    BestModelTracker
)



def train(config_path: str = None, use_wandb: bool = True, wandb_project: str = "ppo-cartpole"):
    """
    Main training function with W&B integration
    
    Args:
        config_path: Path to config file
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
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
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Num envs: {config['environment'].get('num_envs', 1)}")
    print(f"W&B logging: {'Enabled' if use_wandb else 'Disabled'}")
    print("="*70 + "\n")
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(config['training'].get('checkpoint_dir', 'checkpoints'))
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")
    
    # Save config
    save_config(config, checkpoint_dir / 'config.yaml')
    
    # ✅ Initialize W&B
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            config=config,
            name=f"ppo_cartpole_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dir=str(checkpoint_dir),
            save_code=True
        )
        print(f"✓ W&B run initialized: {run.name}")
        print(f"  Dashboard: {run.url}\n")
    
    # Create environments
    print("Creating environments...")
    train_env = create_environment(config['environment']['name'], config['environment'])
    eval_config = config['environment'].copy()
    eval_config['num_envs'] = 1
    eval_env = create_environment(config['environment']['name'], eval_config)
    
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
    
    # ✅ Watch model with W&B
    if use_wandb:
        wandb.watch(agent.network, log='all', log_freq=100)
    
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
        'eval_timesteps': [],
        'losses': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'approx_kls': [],
        'clip_fractions': [],
        'learning_rates': [],
        'best_reward_history': []
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
            action, info = agent.select_action(obs)
            
            if is_vectorized:
                next_obs, rewards, terminated, truncated, env_info = train_env.step(action)
                dones = np.logical_or(terminated, truncated)
                
                agent.buffer.add(
                    obs=obs,
                    action=action,
                    reward=rewards,
                    value=info['value'],
                    log_prob=info['log_prob'],
                    done=dones,
                    raw_action=info['raw_action']
                )
                
                current_episode_rewards += rewards
                for i in range(num_envs):
                    if dones[i]:
                        episode_rewards.append(current_episode_rewards[i])
                        current_episode_rewards[i] = 0
                
                obs = next_obs
            else:
                next_obs, reward, terminated, truncated, env_info = train_env.step(action[0])
                done = terminated or truncated
                
                agent.buffer.add(
                    obs=obs,
                    action=np.array([action[0]]),
                    reward=reward,
                    value=info['value'][0],
                    log_prob=info['log_prob'][0],
                    done=done,
                    raw_action=info['raw_action']
                )
                
                current_episode_rewards[0] += reward
                if done:
                    episode_rewards.append(current_episode_rewards[0])
                    current_episode_rewards[0] = 0
                    obs, _ = train_env.reset()
                else:
                    obs = next_obs
        
        # Update agent
        if is_vectorized:
            _, info = agent.select_action(obs)
            last_values = info['value']
            if isinstance(last_values, np.ndarray) and last_values.ndim > 1:
                last_values = last_values.squeeze()
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
        
        # Compute mean reward
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-20:])
            training_log['mean_rewards'].append(mean_reward)
        else:
            mean_reward = 0.0
            training_log['mean_rewards'].append(0.0)
        
        # ✅ Log to W&B every update
        if use_wandb:
            wandb.log({
                'train/mean_reward': mean_reward,
                'train/loss': metrics['loss'],
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/entropy': metrics['entropy'],
                'train/approx_kl': metrics['approx_kl'],
                'train/clip_fraction': metrics['clip_fraction'],
                'train/learning_rate': metrics['learning_rate'],
                'update': update + 1,
                'timestep': current_timestep
            })
        
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
            training_log['best_reward_history'].append(best_model_tracker.best_reward)
            
            # ✅ Log evaluation to W&B
            if use_wandb:
                wandb.log({
                    'eval/reward': eval_reward,
                    'eval/best_reward': best_model_tracker.best_reward,
                    'update': update + 1,
                    'timestep': current_timestep
                })
            
            if is_best:
                print(f"🎉 NEW BEST MODEL!")
                print(f"   Eval reward: {eval_reward:.2f}")
                print(f"   Saved to: {best_model_tracker.checkpoint_path}")
                
                # ✅ Save best model to W&B
                if use_wandb:
                    wandb.run.summary['best_eval_reward'] = eval_reward
                    wandb.run.summary['best_update'] = update + 1
                    wandb.save(str(best_model_tracker.checkpoint_path))
            else:
                print(f"   Eval reward: {eval_reward:.2f}")
                print(f"   Best so far: {best_model_tracker.best_reward:.2f} "
                      f"(update {best_model_tracker.best_update})")
            
            print(f"{'='*70}\n")
        
        # Print progress
        if (update + 1) % eval_frequency == 0:
            eval_reward_str = f"{eval_reward:.2f}" if eval_reward is not None else "N/A"
            
            print(f"Update {update+1}/{n_updates} | "
                  f"Timesteps: {current_timestep}/{total_timesteps} | "
                  f"Mean Reward: {mean_reward:.2f} | "
                  f"Eval Reward: {eval_reward_str} | "
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
            
            # ✅ Save checkpoint to W&B
            if use_wandb:
                wandb.save(str(checkpoint_path))
    
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
    
    # ✅ Log final results to W&B
    if use_wandb:
        wandb.log({
            'final/mean_reward': final_results['mean_reward'],
            'final/std_reward': final_results['std_reward'],
            'final/min_reward': final_results['min_reward'],
            'final/max_reward': final_results['max_reward']
        })
        wandb.save(str(checkpoint_dir / 'final_model.pt'))
    
    # Save training log
    log_path = checkpoint_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to: {log_path}")
    
    # Save evaluation results
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
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Total updates: {n_updates}")
    print(f"\nBest Model:")
    print(f"  Update: {best_model_tracker.best_update}")
    print(f"  Timestep: {best_model_tracker.best_timestep}")
    print(f"  Eval reward: {best_model_tracker.best_reward:.2f}")
    print(f"\nFinal Model:")
    print(f"  Eval reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    if use_wandb:
        print(f"W&B dashboard: {wandb.run.url}")
    print("="*70 + "\n")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    # ✅ Finish W&B run
    if use_wandb:
        wandb.finish()
    
    return training_log, final_results, checkpoint_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent on CartPole')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config.yaml in same directory)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='ppo-cartpole',
        help='W&B project name (default: ppo-cartpole)'
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project
    )