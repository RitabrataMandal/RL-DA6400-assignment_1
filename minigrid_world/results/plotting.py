import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import os

# Load the saved mean and variance rewards
data = load(os.path.join("results", "sarsa_alpha_0.3_epsilon_0.2_lamda_0.9_episode_1000.npy"), allow_pickle=True).item()
mean_rewards = data['mean']
variance_rewards = data['variance']

# Compute moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_mean = moving_average(mean_rewards)
smoothed_variance = moving_average(variance_rewards)

plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean, label="Smoothed Mean Episodic Return", color='b')
plt.fill_between(range(len(smoothed_mean)), smoothed_mean - smoothed_variance, smoothed_mean + smoothed_variance, color='b', alpha=0.2, label="Variance")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Smoothed Episodic Return vs Episode Number (Moving Average)")
plt.grid()
plt.show()
