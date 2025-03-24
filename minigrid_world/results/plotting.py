import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import os

# Load the saved mean and variance rewards
data = load(os.path.join("results", "sarsa_mean_variance_alpha_0.3_epsilon_0.2_lamda_0.9.npy"), allow_pickle=True).item()
mean_rewards = data['mean']
variance_rewards = data['variance']

plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, label="Mean Episodic Return", color='b')
plt.fill_between(range(len(mean_rewards)), mean_rewards - variance_rewards, mean_rewards + variance_rewards, color='b', alpha=0.2, label="Variance")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Episodic Return vs Episode Number")
plt.grid()
plt.show()
