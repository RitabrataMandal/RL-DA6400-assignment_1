import matplotlib.pyplot as plt
import numpy as np
from sarsa_runner import run_sarsa
from q_runner import run_qlearning

def moving_average(data, window_size=5):
    """Applies a simple moving average to smooth the data."""
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_two_sarsa_results(
    episodes,
    mean1, std1,
    mean2, std2,
    smooth_window=5,
    title="SARSA (CartPole)"
):
    """
    Plots two curves with optional smoothing, shaded std, and a custom title.
    Red = Type 1, Blue = Type 2.
    """
    if smooth_window > 1:
        sm_mean1 = moving_average(mean1, smooth_window)
        sm_std1  = moving_average(std1, smooth_window)
        sm_mean2 = moving_average(mean2, smooth_window)
        sm_std2  = moving_average(std2, smooth_window)
        sm_episodes = episodes[(smooth_window-1)//2 : -(smooth_window//2)] if smooth_window > 1 else episodes
    else:
        sm_mean1, sm_std1 = mean1, std1
        sm_mean2, sm_std2 = mean2, std2
        sm_episodes = episodes

    plt.figure(figsize=(8, 5))

    # --- Type 1 (Red) ---
    plt.plot(sm_episodes, sm_mean1, color='red', label=f'epsilon={epsilon1}, alpha={alpha1}')
    plt.fill_between(
        sm_episodes,
        sm_mean1 - 1*sm_std1,
        sm_mean1 + 1*sm_std1,
        color='red', alpha=0.2
    )

    # --- Type 2 (Blue) ---
    plt.plot(sm_episodes, sm_mean2, color='blue', label=f'epsilon={epsilon2}, alpha={alpha2}')
    plt.fill_between(
        sm_episodes,
        sm_mean2 - 1*sm_std2,
        sm_mean2 + 1*sm_std2,
        color='blue', alpha=0.2
    )

    # Labels, legend, etc.
    plt.title(title)
    plt.xlabel('Episode Number')
    plt.ylabel('Episodic Return')
    plt.legend(loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"plots/sarsa_{alpha1}_{epsilon1}vs{alpha2}_{epsilon2}.png")
if __name__ == "__main__":

    alpha1, gamma1, epsilon1 = 0.05, 0.99, 0.5
    alpha2, gamma2, epsilon2 = 0.2, 0.99, 0.5

    env_name = "CartPole-v1"
    seeds = [100, 200, 300, 400, 500]
    num_episodes = 4000
    num_steps = 500
    num_bins = 20

    rewards_type1 = run_qlearning(env_name, alpha1, gamma1, epsilon1, seeds, num_episodes, num_steps, num_bins)

    rewards_type2 = run_qlearning(env_name, alpha2, gamma2, epsilon2, seeds, num_episodes, num_steps, num_bins)

    mean1 = np.mean(rewards_type1, axis=0)
    std1  = np.std(rewards_type1, axis=0)
    mean2 = np.mean(rewards_type2, axis=0)
    std2  = np.std(rewards_type2, axis=0)

    episodes = np.arange(1, num_episodes + 1)

    # Plot
    plot_two_sarsa_results(
        episodes,
        mean1, std1,
        mean2, std2,
        smooth_window=5,  
        title="SARSA (CartPole)" 
    )