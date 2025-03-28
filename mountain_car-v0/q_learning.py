import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from numpy import save

# Hyperparameters
temperature = 0.10075  # Temperature for softmax
gamma = 0.99  # Discount factor
alpha = 0.47706
episode = 2000  # Number of episodes
bins = 30  
seeds = [100, 200, 300, 400, 500] 

env = gym.make('MountainCar-v0')
n_action = env.action_space.n
env_low = env.observation_space.low
env_high = env.observation_space.high

def getState(state, env_low=env_low, env_high=env_high, bins=bins):
    """Returns the discretized position and velocity of an observation"""
    discretized_env = (env_high - env_low) / bins
    discretized_pos = int((state[0] - env_low[0]) / discretized_env[0])
    discretized_vel = int((state[1] - env_low[1]) / discretized_env[1])
    # Clip to avoid out-of-bound errors
    discretized_pos = np.clip(discretized_pos, 0, bins - 1)
    discretized_vel = np.clip(discretized_vel, 0, bins - 1)
    return discretized_pos, discretized_vel

def softmax(q_values, temperature):
    """Returns a probability distribution over actions using softmax"""
    exp_q = np.exp(q_values / temperature)
    return exp_q / np.sum(exp_q)

def chooseAction(pos, vel, q_table, temperature):
    """Choose action based on Softmax exploration"""
    q_values = q_table[pos][vel]
    action_probs = softmax(q_values, temperature)
    action = np.random.choice(np.arange(len(q_values)), p=action_probs)
    return action

# Store rewards across all seeds
all_rewards = []

for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    
    q_table_qlearn = np.zeros((bins + 1, bins + 1, env.action_space.n))
    rewards_qlearn = []  

    for ep in range(episode):
        current_reward = 0
        done = False
        truncated = False
        state, _ = env.reset(seed=seed)
        pos, vel = getState(state)

        # while not (done or truncated):
        while not done:
            action = chooseAction(pos, vel, q_table_qlearn, temperature)
            next_state, reward, done, truncated, _ = env.step(action)
            next_pos, next_vel = getState(next_state)
            # if done or truncated:
            if done:
                q_table_qlearn[pos][vel][action] += alpha * (reward - q_table_qlearn[pos][vel][action])
            else:
                max_next_q = np.max(q_table_qlearn[next_pos][next_vel])
                q_table_qlearn[pos][vel][action] += alpha * (
                    reward + gamma * max_next_q - q_table_qlearn[pos][vel][action]
                )

            pos, vel = next_pos, next_vel
            current_reward += reward
        
        rewards_qlearn.append(current_reward)
        print(f'Seed {seed} Episode {ep+1}/{episode}, Reward: {current_reward}')

    all_rewards.append(rewards_qlearn)

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
plt.title("Episodic Return vs Episode Number (Q-learning with Softmax)")
plt.legend()
# plt.show()

base_file_name = f"qlearn_softmax_alpha_{alpha}_temp_{temperature}_episode_{episode}.npy"
# os.makedirs("results", exist_ok=True)
save(os.path.join("results", base_file_name), {"mean": mean_rewards, "variance": variance_rewards})

print(f"Results saved in: results/{base_file_name}")
