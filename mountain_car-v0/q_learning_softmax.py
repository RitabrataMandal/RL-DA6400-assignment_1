import gym
import numpy as np
import random

def train_q_learning(temperature, alpha, episode=2000, gamma=0.99, bins=30, seeds=[100, 200, 300, 400, 500]):
    env = gym.make('MountainCar-v0')
    n_action = env.action_space.n
    env_low = env.observation_space.low
    env_high = env.observation_space.high

    def getState(state):
        discretized_env = (env_high - env_low) / bins
        discretized_pos = int((state[0] - env_low[0]) / discretized_env[0])
        discretized_vel = int((state[1] - env_low[1]) / discretized_env[1])
        return np.clip(discretized_pos, 0, bins - 1), np.clip(discretized_vel, 0, bins - 1)

    def softmax(q_values, temp):
        exp_q = np.exp(q_values / temp)
        return exp_q / np.sum(exp_q)

    def chooseAction(pos, vel, q_table, temp):
        q_values = q_table[pos][vel]
        action_probs = softmax(q_values, temp)
        return np.random.choice(np.arange(len(q_values)), p=action_probs)

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

            while not done:
                action = chooseAction(pos, vel, q_table, temperature)
                next_state, reward, done, _, _ = env.step(action)
                next_pos, next_vel = getState(next_state)

                if done:
                    q_table[pos][vel][action] += alpha * (reward - q_table[pos][vel][action])
                else:
                    max_next_q = np.max(q_table[next_pos][next_vel])
                    q_table[pos][vel][action] += alpha * (
                        reward + gamma * max_next_q - q_table[pos][vel][action]
                    )

                pos, vel = next_pos, next_vel
                current_reward += reward

            rewards.append(current_reward)

        all_rewards.append(rewards)

    env.close()

    mean_rewards = np.mean(all_rewards, axis=0)
    all_time_avg_reward = np.mean(mean_rewards)

    return all_time_avg_reward
