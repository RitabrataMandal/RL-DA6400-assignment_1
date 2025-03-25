import argparse
import os
import numpy as np
import gymnasium as gym

from q_table import Qtable
from sarsa_agent import SarasLearner

def run_experiment(alpha, gamma, epsilon, seeds, num_episodes, num_steps, num_bins, algorithm):
    
    performance_list = []
    for seed in seeds:
        env = gym.make('CartPole-v1')
        np.random.seed(seed)
        
        # Initialize Q-table and discretization bins.
        q_table, bins = Qtable(env.observation_space, env.action_space, num_bins)
        
        # Instantiate the SARSA learner.
        if algorithm == "sarsa":
            learner = SarasLearner(alpha, gamma, epsilon, q_table, bins, env, seed)
        else:
            raise ValueError("This tuning script currently supports only sarsa.")
        
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
    parser = argparse.ArgumentParser(description="Hyper-parameter tuning for SARSA (minimizing regret).")
    parser.add_argument("--algorithm", type=str, default="sarsa", choices=["sarsa"],
                        help="Algorithm to use (only sarsa is supported in this tuning script).")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes per experiment run.")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps per episode.")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of bins for state discretization.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[100, 100, 100, 100, 100],
                        help="List of seeds for running multiple experiments.")
    args = parser.parse_args()

    os.makedirs("./results", exist_ok=True)

    alpha_list = [0.05, 0.1, 0.15, 0.2]
    gamma_list = [0.9]
    epsilon_list = [0.05, 0.1, 0.15, 0.2]  

    best_perf = -float("inf")
    best_params = None
    results = {}

    for alpha in alpha_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                print(f"Testing parameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                perf = run_experiment(alpha, gamma, epsilon, args.seeds,
                                      args.num_episodes, args.num_steps, args.num_bins,
                                      args.algorithm)
                results[(alpha, gamma, epsilon)] = perf
                print(f"Average reward over last 100 episodes (averaged across seeds): {perf:.2f}\n")
                if perf > best_perf:
                    best_perf = perf
                    best_params = (alpha, gamma, epsilon)

    print("========================================================")
    print(f"Best hyper-parameters: alpha={best_params[0]}, gamma={best_params[1]}, epsilon={best_params[2]}")
    print(f"Best performance (average reward over last 100 episodes): {best_perf:.2f}")

    results_file = "./results/tuning_results_sarsa.txt"
    with open(results_file, "w") as f:
        f.write("Hyper-parameter tuning results for SARSA\n")
        f.write("Format: (alpha, gamma, epsilon): performance\n")
        for params, perf in results.items():
            f.write(f"{params}: {perf:.2f}\n")
        f.write("\n")
        f.write(f"Best hyper-parameters: alpha={best_params[0]}, gamma={best_params[1]}, epsilon={best_params[2]}\n")
        f.write(f"Best performance: {best_perf:.2f}\n")
    print(f"Tuning results saved to {results_file}")

if __name__ == '__main__':
    main()
