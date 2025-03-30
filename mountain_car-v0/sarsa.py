import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from numpy import save

# Hyperparameters
epsilon = 0.010567 
gamma = 0.99  
alpha = 0.44469  
episode = 2000 
bins = 30  
seeds = [100, 200, 300, 400, 500]  

env = gym.make('MountainCar-v0',render_mode='rgb_array')
n_action = env.action_space.n
env_low = env.observation_space.low
env_high = env.observation_space.high


def getState(state, env_low=env_low, env_high=env_high, bins=bins):
    """Returns the discretized position and velocity of an observation"""
    discretized_env = (env_high - env_low) / bins
    discretized_pos = int((state[0] - env_low[0]) / discretized_env[0])
    discretized_vel = int((state[1] - env_low[1]) / discretized_env[1])
    discretized_pos = np.clip(discretized_pos, 0, bins - 1)
    discretized_vel = np.clip(discretized_vel, 0, bins - 1)
    return discretized_pos, discretized_vel

def chooseAction(pos, vel, q_table, epsilon):
    """Choose action based on an epsilon greedy strategy"""
    if random.random() < epsilon:  
        action = env.action_space.sample()
    else:  
        action = np.argmax(q_table[pos][vel])
    return action

all_rewards = []
for seed in seeds:
    print(f"\n=== Training with Seed: {seed} ===")
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    
    
    q_table_sarsa = np.zeros((bins + 1, bins + 1, env.action_space.n))
    
    rewards_sarsa = []  

    for ep in range(episode):
        current_reward = 0
        done = False
        truncated = False
        state, _ = env.reset(seed=seed)
        pos, vel = getState(state)
        action = chooseAction(pos, vel, q_table_sarsa, epsilon)

        # while not (done or truncated): 
        while not done: 
            next_state, reward, done, truncated, _ = env.step(action)
            next_pos, next_vel = getState(next_state)
            next_action = chooseAction(next_pos, next_vel, q_table_sarsa, epsilon)
    
            if done:
                q_table_sarsa[pos][vel][action] += alpha * (reward - q_table_sarsa[pos][vel][action])
            else:
                q_table_sarsa[pos][vel][action] += alpha * (
                    reward + gamma * q_table_sarsa[next_pos][next_vel][next_action] - q_table_sarsa[pos][vel][action]
                )

            pos, vel = next_pos, next_vel
            action = next_action
            current_reward += reward
        rewards_sarsa.append(current_reward)
        print(f'seed {seed} Episode {ep+1}/{episode}, Reward: {current_reward}')
    all_rewards.append(rewards_sarsa)

env.close()

all_rewards = np.array(all_rewards)

mean_rewards = np.mean(all_rewards, axis=0)
variance_rewards = np.var(all_rewards, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, label="Mean Reward")
plt.fill_between(
    range(episode),
    mean_rewards - np.sqrt(variance_rewards),
    mean_rewards + np.sqrt(variance_rewards),
    color="b",
    alpha=0.2,
    label="Variance"
)
plt.xlabel("Episodes")
plt.ylabel("Return (Smoothed)")
plt.title("Episodic Return vs Episode Number (SARSA with epsilon greedy)")
plt.legend()
# plt.grid()
plt.show()

base_file_name = f"sarsa_alpha_{alpha}_epsilon_{epsilon}_episode_{episode}.npy"
# os.makedirs("results", exist_ok=True)
# save(os.path.join("results", base_file_name), {"mean": mean_rewards, "variance": variance_rewards})

# print(f"Results saved in: results/{base_file_name}")
