import argparse
import gymnasium as gym
import numpy as np
import os
from q_table import Qtable
from q_agent import QLearner
# from sarsa_agent import SarasLearner
from utils import plot_reward_with_variance, render_and_save_frames

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='q_learning', help='Algorithm: q_learning or sarsa.')
parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate.')
parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor.')
parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate (for epsilon-greedy).')
parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes.')
parser.add_argument('--num_steps', type=int, default=200, help='Number of steps per episode.')
parser.add_argument('--num_bins', type=int, default=20, help='Number of bins for discretizing the state space.')
# Instead of a single seed, we allow a list of seeds for averaging
parser.add_argument('--seeds', type=int, nargs='*', default=[100, 100, 100, 100, 100],
                    help='List of seeds for multiple experiments.')
parser.add_argument('--policy', type=str, default='epsilon', choices=['epsilon', 'softmax'],help='Exploration policy: "epsilon" or "softmax".')
parser.add_argument('--temperature', type=float, default=1.0,help='Temperature parameter for the softmax policy (if used).')
args = parser.parse_args()

# Ensure output directories exist
os.makedirs("./plots", exist_ok=True)
os.makedirs("./videos", exist_ok=True)

# base_file_name = (f"{args.algorithm}_alpha_{args.alpha}_gamma_{args.gamma}_epsilon_{args.epsilon}_"
#                   f"num_episodes_{args.num_episodes}_num_steps_{args.num_steps}_num_bins_{args.num_bins}")
base_file_name = (f"{args.algorithm}_alpha_{args.alpha}_temperature_{args.temperature}")
print("Base file name:", base_file_name)

all_rewards = []

for seed in args.seeds:
    print(f"\nRunning experiment with seed: {seed}")
    env = gym.make('MountainCar-v0')
    np.random.seed(seed)
    
    q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)
    
    if args.algorithm == "q_learning":
        learner = QLearner(args.alpha, args.gamma, args.temperature, q_table, bins, env, seed)
    elif args.algorithm == "sarsa":
        learner = SarasLearner(args.alpha, args.gamma, args.epsilon, q_table, bins, env, seed)
    else:
        raise ValueError("The algorithm should be either q_learning or sarsa.")
    
    reward_list = learner.learn(args.num_episodes, args.num_steps)
    all_rewards.append(reward_list)
    
    env.close()

all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

plot_reward_with_variance(mean_rewards, std_rewards,args.algorithm, args.policy,args.num_episodes, "./plots/", base_file_name + ".png")
