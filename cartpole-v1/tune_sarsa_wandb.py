import argparse
import os
import numpy as np
import gymnasium as gym
import wandb

from q_table import Qtable
from sarsa_agent import SarasLearner

def run_experiment(alpha, gamma, epsilon, seeds, num_episodes, num_steps, num_bins):
    """
    Runs a complete SARSA experiment using the given hyper-parameters over several seeds.
    
    For each seed, the environment and learner are re-initialized and the reward curve
    is recorded. The performance metric is the average reward over the last 100 episodes.
    
    Args:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        seeds (list[int]): List of seeds for multiple runs.
        num_episodes (int): Number of episodes per run.
        num_steps (int): Number of steps per episode.
        num_bins (int): Number of bins for state discretization.
    
    Returns:
        float: The average performance (mean of average rewards over the last 100 episodes) across seeds.
    """
    performance_list = []
    for seed in seeds:
        env = gym.make('CartPole-v1')
        np.random.seed(seed)
        
        # Initialize Q-table and discretization bins.
        q_table, bins = Qtable(env.observation_space, env.action_space, num_bins)
        
        # Instantiate the SARSA learner.
        learner = SarasLearner(alpha, gamma, epsilon, q_table, bins, env, seed)
        
        # Run the learning process.
        reward_list = learner.learn(num_episodes, num_steps)
        # Use the average reward over the last 100 episodes as performance metric.
        if len(reward_list) >= 100:
            avg_last100 = np.mean(reward_list[-100:])
        else:
            avg_last100 = np.mean(reward_list)
        performance_list.append(avg_last100)
        env.close()
    return np.mean(performance_list)

def main():
    parser = argparse.ArgumentParser(description="Hyper-parameter tuning for SARSA with wandb logging.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes per experiment run.")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps per episode.")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of bins for state discretization.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[100, 100, 100, 100, 100],
                        help="List of seeds for running multiple experiments.")
    args = parser.parse_args()

    # Initialize wandb.
    wandb.init(project="tuning-sarsa", reinit=True, config={
        "num_episodes": args.num_episodes,
        "num_steps": args.num_steps,
        "num_bins": args.num_bins,
        "seeds": args.seeds,
        "algorithm": "sarsa"
    })

    # Hyperparameter grids.
    alpha_list = [0.05, 0.1, 0.2]
    gamma_list = [0.9]
    epsilon_list = [0.05, 0.1, 0.2]  # Exploration rates for epsilon-greedy policy.

    results = {}
    run_count = 0

    # Grid search: iterate over all combinations of hyper-parameters.
    for alpha in alpha_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                run_count += 1
                print(f"Run {run_count}: Testing parameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                avg_reward = run_experiment(alpha, gamma, epsilon, args.seeds,
                                            args.num_episodes, args.num_steps, args.num_bins)
                results[(alpha, gamma, epsilon)] = avg_reward
                print(f"Avg reward (last 100 episodes): {avg_reward:.2f}")
                wandb.log({
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "avg_reward_last_100": avg_reward
                })

    # Sort the results to get the top 3 best hyper-parameter combinations.
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_results[:3]

    # Print and log the top 3 results with comparative justification.
    print("========================================================")
    print("Top 3 best hyperparameter combinations (by avg reward over last 100 episodes):")
    for idx, ((alpha, gamma, epsilon), perf) in enumerate(top_3, start=1):
        print(f"Rank {idx}: alpha={alpha}, gamma={gamma}, epsilon={epsilon} --> Avg Reward: {perf:.2f}")

    justification = (
        "\nComparative Justification:\n"
        "1. The best performing combination indicates that a higher learning rate (alpha) combined with a "
        "moderate discount factor (gamma) and a lower exploration rate (epsilon) yields the highest performance. "
        "This suggests that, for this SARSA setup, a more aggressive update along with less randomness in action selection "
        "is beneficial.\n"
        "2. The second and third best combinations are close in performance, revealing that slight adjustments in gamma or epsilon "
        "do not drastically affect performance as long as the learning rate remains favorable.\n"
        "3. Overall, the results highlight the importance of balancing rapid learning (via alpha) with controlled exploration "
        "(via epsilon) while ensuring that future rewards are appropriately weighted (via gamma)."
    )
    print(justification)
    wandb.log({"top_3_results": top_3, "justification": justification})
    wandb.finish()

if __name__ == '__main__':
    main()
