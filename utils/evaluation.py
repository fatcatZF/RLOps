import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an RL agent on an environment
    
    Supports both single and vectorized environments (Gymnasium API).
    For vectorized environments, runs until num_episodes total episodes 
    are completed across all environments.
    
    Properly handles:
    - Gymnasium API (terminated, truncated separate)
    - VectorEnv autoreset behavior
    - Batched observations, rewards, terminations, truncations
    - Dictionary infos from VectorEnv
    
    Uses torch.no_grad() context for memory efficiency and faster inference.
    
    Args:
        agent: RL agent with select_action() method
        env: Environment (single or vectorized)
        num_episodes: Total number of episodes to evaluate
        max_steps_per_episode: Maximum steps per episode (safety limit)
        deterministic: If True, use deterministic actions (default: True)
        render: If True, render the environment (only for single env)
        verbose: If True, print progress and results
    
    Returns:
        Dictionary containing:
            - episode_rewards: List of episode returns
            - episode_lengths: List of episode lengths
            - mean_reward: Mean episode return
            - std_reward: Std of episode returns
            - min_reward: Minimum episode return
            - max_reward: Maximum episode return
            - mean_length: Mean episode length
            - std_length: Std of episode lengths
            - success_rate: Success rate (if env provides 'success' in info)
    
    Examples:
        >>> # Single environment
        >>> from rl_agents import PPOAgent
        >>> from environments import create_environment
        >>> 
        >>> agent = PPOAgent(config)
        >>> agent.load('checkpoints/ppo_100k.pt')
        >>> env = create_environment('cartpole', {'wind_mag': 0.0})
        >>> results = evaluate_agent(agent, env, num_episodes=100)
        >>> print(f"Mean reward: {results['mean_reward']:.2f}")
        >>> 
        >>> # Vectorized environment (Gymnasium VectorEnv)
        >>> import gymnasium as gym
        >>> env_vec = gym.make_vec('CartPole-v1', num_envs=8)
        >>> results = evaluate_agent(agent, env_vec, num_episodes=100)
    """
    # Set agent to evaluation mode
    agent.eval_mode()
    
    # Detect if environment is vectorized
    # Gymnasium VectorEnv has num_envs attribute
    is_vectorized = hasattr(env, 'num_envs') and env.num_envs > 1
    num_envs = env.num_envs if is_vectorized else 1
    
    # Storage for results
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    # Per-environment tracking
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)
    episodes_completed = 0
    
    # Reset environment (Gymnasium API)
    if is_vectorized:
        # VectorEnv returns: observations, infos (both batched)
        obs, infos = env.reset()  # obs: [num_envs, *obs_shape]
    else:
        # Single env returns: observation, info
        obs, info = env.reset()
    
    step = 0
    
    # Use torch.no_grad() for evaluation
    with torch.no_grad():
        while episodes_completed < num_episodes and step < max_steps_per_episode * num_episodes:
            # Render if requested (only for single env)
            if render and not is_vectorized:
                env.render()
            
            # Select action (deterministic for evaluation)
            action, agent_info = agent.select_action(obs, deterministic=deterministic)
            
            # Step environment
            if is_vectorized:
                # VectorEnv step returns: observations, rewards, terminated, truncated, infos
                # All are batched: [num_envs] for scalars, [num_envs, ...] for observations
                next_obs, rewards, terminated, truncated, infos = env.step(action)
                
                # Combine terminated and truncated into done array
                dones = np.logical_or(terminated, truncated)
                
                # Update tracking for each environment
                current_rewards += rewards
                current_lengths += 1
                
                # Check for completed episodes
                # VectorEnv uses autoreset, so we just need to record when episodes finish
                for i in range(num_envs):
                    if dones[i]:
                        # Episode finished in env i (auto-resets automatically)
                        if episodes_completed < num_episodes:
                            episode_rewards.append(float(current_rewards[i]))
                            episode_lengths.append(int(current_lengths[i]))
                            
                            # Check for success (if available)
                            # In VectorEnv, infos is a dict where each key has [num_envs] values
                            if 'success' in infos:
                                episode_successes.append(bool(infos['success'][i]))
                            # Alternative: infos might be a list of dicts (depends on implementation)
                            elif isinstance(infos, list) and 'success' in infos[i]:
                                episode_successes.append(bool(infos[i]['success']))
                            
                            episodes_completed += 1
                            
                            if verbose and episodes_completed % max(1, num_episodes // 10) == 0:
                                print(f"Completed {episodes_completed}/{num_episodes} episodes | "
                                      f"Latest reward: {current_rewards[i]:.2f}")
                        
                        # Reset tracking for this environment
                        # (Environment auto-resets, so we just reset our counters)
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                
                obs = next_obs
            
            else:
                # Single environment (Gymnasium API)
                next_obs, reward, terminated, truncated, info = env.step(action[0])
                
                # Combine terminated and truncated into done
                done = terminated or truncated
                
                # Update tracking
                current_rewards[0] += reward
                current_lengths[0] += 1
                
                # Check if episode finished
                if done:
                    # Episode finished
                    episode_rewards.append(float(current_rewards[0]))
                    episode_lengths.append(int(current_lengths[0]))
                    
                    # Check for success
                    if 'success' in info:
                        episode_successes.append(bool(info['success']))
                    
                    episodes_completed += 1
                    
                    if verbose and episodes_completed % max(1, num_episodes // 10) == 0:
                        print(f"Completed {episodes_completed}/{num_episodes} episodes | "
                              f"Reward: {current_rewards[0]:.2f}")
                    
                    # Reset tracking
                    current_rewards[0] = 0
                    current_lengths[0] = 0
                    
                    # Reset environment for next episode
                    if episodes_completed < num_episodes:
                        next_obs, info = env.reset()
                
                obs = next_obs
            
            step += 1
    
    # Set agent back to training mode
    agent.train_mode()
    
    # Compute statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    
    results = {
        'episode_rewards': episode_rewards.tolist(),
        'episode_lengths': episode_lengths.tolist(),
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'num_episodes': len(episode_rewards)
    }
    
    # Add success rate if available
    if len(episode_successes) > 0:
        results['success_rate'] = float(np.mean(episode_successes))
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of episodes: {results['num_episodes']}")
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Min/Max reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print(f"Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        if 'success_rate' in results:
            print(f"Success rate: {results['success_rate']*100:.1f}%")
        print("="*60 + "\n")
    
    return results


def evaluate_agent_during_training(
    agent,
    env,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000
) -> float:
    """
    Quick evaluation during training (returns only mean reward)
    
    Useful for monitoring training progress without verbose output.
    Uses torch.no_grad() for efficiency.
    
    Args:
        agent: RL agent
        env: Environment (single or vectorized)
        num_episodes: Number of episodes to evaluate
        max_steps_per_episode: Maximum steps per episode
    
    Returns:
        Mean episode reward
    
    Example:
        >>> for update in range(num_updates):
        ...     # Training
        ...     metrics = agent.update(last_value)
        ...     
        ...     # Periodic evaluation
        ...     if update % 50 == 0:
        ...         mean_reward = evaluate_agent_during_training(agent, eval_env)
        ...         print(f"Eval reward: {mean_reward:.2f}")
    """
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=True,
        render=False,
        verbose=False
    )
    return results['mean_reward']


def save_evaluation_results(
    results: Dict[str, Any],
    save_path: str,
    agent_name: str = "agent",
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save evaluation results to file
    
    Args:
        results: Results dictionary from evaluate_agent()
        save_path: Path to save results (e.g., 'results/eval_cartpole.json')
        agent_name: Name of the agent
        additional_info: Additional information to save (e.g., config, timestamp)
    
    Example:
        >>> results = evaluate_agent(agent, env, num_episodes=100)
        >>> save_evaluation_results(
        ...     results,
        ...     'results/eval_cartpole.json',
        ...     agent_name='PPO',
        ...     additional_info={'config': config, 'timesteps': 100000}
        ... )
    """
    import json
    from datetime import datetime
    
    # Create directory if it doesn't exist
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    data = {
        'agent_name': agent_name,
        'timestamp': datetime.now().isoformat(),
        'evaluation_results': results
    }
    
    # Add additional info if provided
    if additional_info is not None:
        data['additional_info'] = additional_info
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Evaluation results saved to: {save_path}")


def compare_agents(
    agents: List[Tuple[str, Any]],
    env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple agents on the same environment
    
    Args:
        agents: List of (name, agent) tuples
        env: Environment to evaluate on
        num_episodes: Number of episodes per agent
        max_steps_per_episode: Maximum steps per episode
    
    Returns:
        Dictionary mapping agent names to their results
    
    Example:
        >>> agent1 = PPOAgent(config1)
        >>> agent1.load('checkpoints/ppo_default.pt')
        >>> agent2 = PPOAgent(config2)
        >>> agent2.load('checkpoints/ppo_tuned.pt')
        >>> 
        >>> comparison = compare_agents(
        ...     agents=[('PPO-Default', agent1), ('PPO-Tuned', agent2)],
        ...     env=env,
        ...     num_episodes=100
        ... )
        >>> 
        >>> for name, results in comparison.items():
        ...     print(f"{name}: {results['mean_reward']:.2f}")
    """
    print("\n" + "="*60)
    print("COMPARING AGENTS")
    print("="*60)
    
    all_results = {}
    
    for name, agent in agents:
        print(f"\nEvaluating: {name}")
        print("-" * 60)
        
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=True,
            verbose=True
        )
        
        all_results[name] = results
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Agent':<20} {'Mean Reward':<15} {'Std':<10} {'Success Rate':<15}")
    print("-" * 60)
    
    for name, results in all_results.items():
        success_str = f"{results['success_rate']*100:.1f}%" if 'success_rate' in results else "N/A"
        print(f"{name:<20} {results['mean_reward']:<15.2f} "
              f"{results['std_reward']:<10.2f} {success_str:<15}")
    
    print("="*60 + "\n")
    
    return all_results






