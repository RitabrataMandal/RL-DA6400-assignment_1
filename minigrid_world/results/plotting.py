import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import os

# Load the saved mean and variance rewards
data1 = load(os.path.join("results", "sarsa_alpha_0.15_epsilon_0.01_episodes_1000.npy"), allow_pickle=True).item()
data2 = load(os.path.join("results", "sarsa_alpha_0.15_epsilon_0.015_episodes_1000.npy"), allow_pickle=True).item()
#best alpgha=.15 epsilon=.015
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
plt.plot(smoothed_mean1, label="alpha=.15,epsilon=.01", color='b')
plt.fill_between(range(len(smoothed_mean1)), smoothed_mean1 - smoothed_variance1, smoothed_mean1 + smoothed_variance1, color='b', alpha=0.2)
plt.plot(smoothed_mean2, label="alpha=.15,epsilon=.015", color='r')
plt.fill_between(range(len(smoothed_mean2)), smoothed_mean2 - smoothed_variance2, smoothed_mean2 + smoothed_variance2, color='r', alpha=0.2)



plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("SARSA(MiniGrid-Dynamic-Obstacles-Random-5x5-v0)")
# plt.grid()
plt.show()
