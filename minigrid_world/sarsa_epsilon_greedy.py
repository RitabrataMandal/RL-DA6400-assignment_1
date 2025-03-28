import gym
import numpy as np 
import matplotlib.pyplot as plt
from numpy import save
from gym_minigrid.wrappers import *
import os

def take_action(q_value, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)  # Explore
    return np.argmax(q_value)  # Exploit

# Hyperparameters
seeds = [100, 200, 300, 400, 500]
episodes = 2000
epsilon = 0.022713
gamma = 0.99
alpha = 0.12801

all_rewards = []

for seed in seeds:
    env = gym.make('MiniGrid-Dynamic-Obstacles-Random-5x5-v0')
    env.reset(seed=seed)
    q_value = np.zeros((3, 25, 4, 2))  # 5x5 Grid
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
            
            # SARSA update
            q_value[action, x1, x2, x3] += alpha * (reward + gamma * q_value[new_action, new_x1, new_x2, new_x3] - q_value[action, x1, x2, x3])
            
            # Move to next state
            x1, x2, x3, action = new_x1, new_x2, new_x3, new_action
            total_reward[ep] += reward
        
        print(f"Seed: {seed} Episode: {ep+1} Reward: {total_reward[ep]}")
    
    all_rewards.append(total_reward)
    env.close()

# Save and plot results
all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
variance_rewards = np.var(all_rewards, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, label="Mean Episodic Return", color='b')
plt.fill_between(range(episodes), mean_rewards - variance_rewards, mean_rewards + variance_rewards, color='b', alpha=0.2, label="Variance")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Episodic Return vs Episode Number (SARSA)")
plt.grid()
# plt.show()

base_file_name = f"sarsa_alpha_{alpha}_epsilon_{epsilon}_episodes_{episodes}.npy"
# os.makedirs("results", exist_ok=True)
save(os.path.join("results", base_file_name), {'mean': mean_rewards, 'variance': variance_rewards})
