import gym
import numpy as np 
import os
from gym_minigrid.wrappers import *

def softmax_action(q_value, temperature):
    """Return an action selected via softmax exploration."""
    exp_values = np.exp(q_value / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(q_value), p=probabilities)

def train_qlearning_softmax(temperature, alpha, episodes=1000, gamma=0.99, seeds=[100, 200, 300, 400, 500]):
    all_rewards = []

    for seed in seeds:
        env = gym.make('MiniGrid-Dynamic-Obstacles-Random-5x5-v0')
        env.reset(seed=seed)
        # Q-table dimensions: actions x (5x5 grid) x agent direction (4) x front cell flag (2)
        q_value = np.zeros((3, 25, 4, 2))
        total_reward = np.zeros(episodes)
        
        for ep in range(episodes):
            env.reset()
            terminated, truncated = False, False
            # Compute state indices
            x1 = env.agent_pos[0] * 5 + env.agent_pos[1]
            x2 = env.agent_dir
            front_cell = env.grid.get(*env.front_pos)
            x3 = 1 if (front_cell and front_cell.type != "goal") else 0
            
            # Select action using softmax
            action = softmax_action(q_value[:, x1, x2, x3], temperature)
            
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(action)
                new_x1 = env.agent_pos[0] * 5 + env.agent_pos[1]
                new_x2 = env.agent_dir
                front_cell = env.grid.get(*env.front_pos)
                new_x3 = 1 if (front_cell and front_cell.type != "goal") else 0
                
                # Q-learning update with softmax exploration
                best_next_action = np.argmax(q_value[:, new_x1, new_x2, new_x3])
                q_value[action, x1, x2, x3] += alpha * (
                    reward + gamma * q_value[best_next_action, new_x1, new_x2, new_x3] - q_value[action, x1, x2, x3]
                )
                
                # Move to next state
                x1, x2, x3 = new_x1, new_x2, new_x3
                action = softmax_action(q_value[:, x1, x2, x3], temperature)
                total_reward[ep] += reward
            
            print(f"Seed: {seed} Episode: {ep+1} Reward: {total_reward[ep]}")
        
        all_rewards.append(total_reward)
        env.close()

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    all_time_avg_reward = np.mean(mean_rewards)
    return all_time_avg_reward, mean_rewards
