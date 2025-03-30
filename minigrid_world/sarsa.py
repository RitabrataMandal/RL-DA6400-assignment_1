import gym
import numpy as np 
import os
from gym_minigrid.wrappers import *
import random

def take_action(q_value, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)  
    return np.argmax(q_value)  

def train_minigrid(epsilon, alpha, episodes=2000, gamma=0.99, seeds=[100, 200, 300, 400, 500]):
    
    all_rewards = []

    for seed in seeds:
        env = gym.make('MiniGrid-Dynamic-Obstacles-Random-5x5-v0')
        env.reset(seed=seed)
        # Q-table dimensions: action x (5x5 grid) x agent direction (4) x front cell flag (2)
        q_value = np.zeros((3, 25, 4, 2))
        total_reward = np.zeros(episodes)

        for ep in range(episodes):
            env.reset()
            terminated, truncated = False, False
            x1 = env.agent_pos[0] * 5 + env.agent_pos[1]
            x2 = env.agent_dir
            front_cell = env.grid.get(*env.front_pos)
            x3 = 1 if (front_cell and front_cell.type != "goal") else 0

            action = take_action(q_value[:, x1, x2, x3], epsilon)

            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(action)
                new_x1 = env.agent_pos[0] * 5 + env.agent_pos[1]
                new_x2 = env.agent_dir
                front_cell = env.grid.get(*env.front_pos)
                new_x3 = 1 if (front_cell and front_cell.type != "goal") else 0
                new_action = take_action(q_value[:, new_x1, new_x2, new_x3], epsilon)

                q_value[action, x1, x2, x3] += alpha * (
                    reward + gamma * q_value[new_action, new_x1, new_x2, new_x3] - q_value[action, x1, x2, x3]
                )

                x1, x2, x3, action = new_x1, new_x2, new_x3, new_action
                total_reward[ep] += reward

            print(f"Seed: {seed} Episode: {ep+1} Reward: {total_reward[ep]}")

        all_rewards.append(total_reward)
        env.close()

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    all_time_avg_reward = np.mean(mean_rewards)
    return all_time_avg_reward, mean_rewards  