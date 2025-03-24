import gym
import numpy as np 
import matplotlib.pyplot as plt
from numpy import save
from gym_minigrid.wrappers import *
import os

def take_action(q_value, epsilon):
    prob = np.random.random()
    if prob < epsilon:
        return np.random.randint(0, 3)
    else:
        return np.argmax(q_value)

seeds = [100, 200, 300, 400, 500]
episode = 1000
epsilon = 0.2
gamma = 0.9
alpha = 0.3
lamda = 0.9

all_rewards = []

for seed in seeds:
    env = gym.make('MiniGrid-Dynamic-Obstacles-Random-5x5-v0')
    env.reset(seed=seed)
    q_value = np.zeros((3, 16, 4, 2))
    total_reward = np.zeros(episode)
    
    for K in range(episode):
        env.reset()
        terminated = False
        truncated = False
        front_cell = env.grid.get(*env.front_pos)
        not_clear = front_cell and front_cell.type != "goal"
        x3 = 1 if not_clear else 0
        x1 = (env.agent_pos[0] - 1) * 4 + (env.agent_pos[1] - 1)
        x2 = env.agent_dir
        action = take_action(q_value[:, x1, x2, x3], epsilon)
        
        visited_states = []
        E = np.zeros((3, 36, 4, 2))
        
        while not (terminated or truncated):
            x1 = (env.agent_pos[0] - 1) * 4 + (env.agent_pos[1] - 1)
            x2 = env.agent_dir
            E[action, x1, x2, x3] = 1
            observation, reward, terminated, truncated, info = env.step(action)
            
            if [action, x1, x2, x3] not in visited_states:
                visited_states.append([action, x1, x2, x3])
            
            new_x1 = (env.agent_pos[0] - 1) * 4 + (env.agent_pos[1] - 1)
            new_x2 = env.agent_dir
            front_cell = env.grid.get(*env.front_pos)
            not_clear = front_cell and front_cell.type != "goal"
            new_x3 = 1 if not_clear else 0
            
            new_action = take_action(q_value[:, new_x1, new_x2, new_x3], epsilon)
            
            for state in visited_states:
                error = (reward + gamma * (q_value[new_action, new_x1, new_x2, new_x3]) - q_value[action, x1, x2, x3])
                q_value[state[0], state[1], state[2], state[3]] += alpha * (E[state[0], state[1], state[2], state[3]]) * error
            
            action = new_action
            x3 = new_x3
            E = gamma * lamda * E
            total_reward[K] += reward
        
        print(f"Seed: {seed} Episode: {K+1} Reward: {total_reward[K]}")
    
    all_rewards.append(total_reward)
    env.close()

all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
variance_rewards = np.var(all_rewards, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, label="Mean Episodic Return", color='b')
plt.fill_between(range(episode), mean_rewards - variance_rewards, mean_rewards + variance_rewards, color='b', alpha=0.2, label="Variance")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Episodic Return vs Episode Number")
plt.grid()
plt.show()

base_file_name = f"sarsa_alpha_{alpha}_epsilon_{epsilon}_lamda_{lamda}.npy"
os.makedirs("results", exist_ok=True)
save(os.path.join("results", base_file_name), {'mean': mean_rewards, 'variance': variance_rewards})
