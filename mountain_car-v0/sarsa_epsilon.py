import gym
import numpy as np
import random

def train_sarsa(epsilon, alpha, episode=2000, gamma=0.99, bins=30, seeds=[100, 200, 300, 400, 500]):
    """
    Runs SARSA with epsilon-greedy exploration on the MountainCar-v0 environment.
    Returns the all-time average mean reward across episodes.
    """
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env_low = env.observation_space.low
    env_high = env.observation_space.high

    def getState(state):
        discretized_env = (env_high - env_low) / bins
        discretized_pos = int((state[0] - env_low[0]) / discretized_env[0])
        discretized_vel = int((state[1] - env_low[1]) / discretized_env[1])
        return np.clip(discretized_pos, 0, bins - 1), np.clip(discretized_vel, 0, bins - 1)

    def chooseAction(pos, vel, q_table, epsilon):
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[pos][vel])  # Exploit
        return action

    all_rewards = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)
        
        q_table = np.zeros((bins + 1, bins + 1, env.action_space.n))
        rewards = []

        for ep in range(episode):
            current_reward = 0
            done = False
            state, _ = env.reset(seed=seed)
            pos, vel = getState(state)
            action = chooseAction(pos, vel, q_table, epsilon)

            while not done:
                next_state, reward, done, truncated, _ = env.step(action)
                next_pos, next_vel = getState(next_state)
                next_action = chooseAction(next_pos, next_vel, q_table, epsilon)

                # SARSA update rule
                if done:
                    q_table[pos][vel][action] += alpha * (reward - q_table[pos][vel][action])
                else:
                    q_table[pos][vel][action] += alpha * (
                        reward + gamma * q_table[next_pos][next_vel][next_action] - q_table[pos][vel][action]
                    )

                pos, vel = next_pos, next_vel
                action = next_action
                current_reward += reward

            rewards.append(current_reward)
        all_rewards.append(rewards)

    env.close()

    # Calculate mean rewards across seeds (per episode)
    mean_rewards = np.mean(all_rewards, axis=0)
    all_time_avg_reward = np.mean(mean_rewards)
    return all_time_avg_reward
