import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import os

# Load the saved mean and variance rewards
# data1 = load(os.path.join("results", "q_learning_softmax_alpha_0.1_tau_0.9_episodes_10000.npy"), allow_pickle=True).item()
# data2 = load(os.path.join("results", "q_learning_softmax_alpha_0.5_tau_1_episodes_10000.npy"), allow_pickle=True).item()

data1 = load(os.path.join("results", "sarsa_alpha_0.1_epsilon_0.1_episode_5000.npy"), allow_pickle=True).item()
data2 = load(os.path.join("results", "sarsa_alpha_0.1_epsilon_0.01_episode_5000.npy"), allow_pickle=True).item()
#best alpgha=.15 epsilon=.015 for SARSA

mean_rewards1 = data1['mean']
variance_rewards1 = data1['variance']

mean_rewards2 = data2['mean']
variance_rewards2 = data2['variance']

# Compute moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_mean1 = moving_average(mean_rewards1)
smoothed_variance1 = moving_average(variance_rewards1)

smoothed_mean2 = moving_average(mean_rewards2)
smoothed_variance2 = moving_average(variance_rewards2)


plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean1, label="alpha=.1,epsilon=.1", color='b')
# plt.plot(smoothed_mean1, label="alpha=.1,tau=.9", color='b')
plt.fill_between(range(len(smoothed_mean1)), smoothed_mean1 - smoothed_variance1, smoothed_mean1 + smoothed_variance1, color='b', alpha=0.2)
plt.plot(smoothed_mean2, label="alpha=.1,epsilon=.01", color='r')
# plt.plot(smoothed_mean1, label="alpha=.5,tau=1", color='r')
plt.fill_between(range(len(smoothed_mean2)), smoothed_mean2 - smoothed_variance2, smoothed_mean2 + smoothed_variance2, color='r', alpha=0.2)



plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("SARSA(MountainCar-v0)")
# plt.title("Q-Learning(MountainCar-v0)")
# plt.grid()
plt.show()
