import gymnasium as gym
import numpy as np
from sarsa_agent import SarasLearner
from q_table import Qtable

def run_sarsa(env_name, alpha, gamma, epsilon, seeds, num_episodes, num_steps, num_bins):
    """
    Runs SARSA on the given environment for multiple seeds.
    
    Args:
        env_name (str): Name of the gym environment (e.g. "CartPole-v1").
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        seeds (list[int]): List of seeds to run.
        num_episodes (int): Number of episodes per seed.
        num_steps (int): Max steps per episode.
        num_bins (int): Number of bins for discretizing state space.

    Returns:
        np.ndarray: A 2D array of shape (len(seeds), num_episodes) with the episodic returns.
    """
    all_rewards = []
    for seed in seeds:
        # Create environment
        env = gym.make(env_name)
        np.random.seed(seed)
        
        # Create Q-table
        q_table, bins = Qtable(env.observation_space, env.action_space, num_bins)
        
        # Create SARSA learner
        sarsa_learner = SarasLearner(alpha, gamma, epsilon, q_table, bins, env, seed)
        
        # Learn
        reward_list = sarsa_learner.learn(num_episodes, num_steps)
        all_rewards.append(reward_list)
        
        env.close()
    
    return np.array(all_rewards)  # shape: [num_seeds, num_episodes]
