import gymnasium as gym
import numpy as np
from sarsa_agent import SarasLearner
from q_table import Qtable

def run_sarsa(env_name, alpha, gamma, epsilon, seeds, num_episodes, num_steps, num_bins):

    all_rewards = []
    for seed in seeds:
        env = gym.make(env_name)
        np.random.seed(seed)
        
        q_table, bins = Qtable(env.observation_space, env.action_space, num_bins)
        
        sarsa_learner = SarasLearner(alpha, gamma, epsilon, q_table, bins, env, seed)
        
        reward_list = sarsa_learner.learn(num_episodes, num_steps)
        all_rewards.append(reward_list)
        
        env.close()
    
    return np.array(all_rewards)  
