import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import re
import os

episode =2000
alpha1=0.12704
alpha2=0.2107
temp1 =0.1
temp2=0.1
epsilon1=0.02681
epsilon2=0.024791
# file1=f"qlearn_softmax_alpha_{alpha1}_temp_{temp1}_episode_{episode}.npy"
# file2=f"qlearn_softmax_alpha_{alpha2}_temp_{temp2}_episode_{episode}.npy"
file3=f"sarsa_alpha_{alpha1}_epsilon_{epsilon1}_episode_{episode}.npy"
file4=f"sarsa_alpha_{alpha2}_epsilon_{epsilon2}_episode_{episode}.npy"

# Load the saved mean and variance rewards
# data1 = load(os.path.join("results", file1 ), allow_pickle=True).item()
# data2 = load(os.path.join("results", file2), allow_pickle=True).item()

data1 = load(os.path.join("results",file3), allow_pickle=True).item()
data2 = load(os.path.join("results", file4), allow_pickle=True).item()



mean_rewards1 = data1['mean']
# print(mean_rewards1)
print(f"alpha:{alpha1} tau:{temp1}::{np.mean(mean_rewards1[-100:])}")
variance_rewards1 = data1['variance']

mean_rewards2 = data2['mean']
print(f"alpha:{alpha2} tau:{temp2}::{np.mean(mean_rewards2[-100:])}")
variance_rewards2 = data2['variance']

# Compute moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_mean1 = moving_average(mean_rewards1)
smoothed_variance1 = moving_average(variance_rewards1)

smoothed_mean2 = moving_average(mean_rewards2)
smoothed_variance2 = moving_average(variance_rewards2)


plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean1, label=f"alpha={alpha1},epsilon={epsilon1}", color='b')
# plt.plot(smoothed_mean1, label=f"alpha={alpha1},tau={temp1}", color='b')
plt.fill_between(range(len(smoothed_mean1)), smoothed_mean1 - np.sqrt(smoothed_variance1), smoothed_mean1 + np.sqrt(smoothed_variance1), color='b', alpha=0.2)
plt.plot(smoothed_mean2, label=f"alpha={alpha2},epsilon={epsilon2}", color='r')
# plt.plot(smoothed_mean2, label=f"alpha={alpha2},tau={temp2}", color='r')
plt.fill_between(range(len(smoothed_mean2)), smoothed_mean2 - np.sqrt(smoothed_variance2), smoothed_mean2 + np.sqrt(smoothed_variance2), color='r', alpha=0.2)



plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("SARSA(MountainCar-v0)")
# plt.title("Q-Learning(MountainCar-v0)")
# plt.title("SARSA vs Q-Learning")
# plt.grid()
plt.show()
