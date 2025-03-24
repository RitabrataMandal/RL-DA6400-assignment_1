import gymnasium as gym
import numpy as np
from q_agent_m import QLearner
from q_table_m import Qtable

def run_qlearning(env_name, alpha, gamma, temperature, seeds, num_episodes, num_steps, num_bins):
    all_rewards = []
    
    for seed in seeds:
        env = gym.make(env_name)
        np.random.seed(seed)

        # Assuming Qtable is a class and should be instantiated
        q_table, bins = Qtable(env.observation_space, env.action_space, num_bins)

        # Corrected parameter passing
        q_learner = QLearner(alpha, gamma, temperature, q_table=q_table,bins=bins,env= env, seed=seed)

        # Use the provided num_episodes and num_steps
        reward_list = q_learner.learn(num_episodes, num_steps)
        all_rewards.append(reward_list)

        env.close()

    return np.array(all_rewards)
