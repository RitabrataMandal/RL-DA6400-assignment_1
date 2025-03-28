import wandb
import yaml
import numpy as np
from sarsa_runner import run_sarsa

with open("sweep_sarsa.yaml","r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="SARSA-cartpole")

def train():
    run = wandb.init()
    config= wandb.config

    gamma =0.99
    env_name = "CartPole-v1"
    seeds = [100, 200, 300, 400, 500]
    num_episodes = 2000
    num_steps = 500
    num_bins = 20

    rewards = run_sarsa(
        env_name,
        config.alpha,
        gamma,
        config.epsilon,
        seeds,
        num_episodes,
        num_steps,
        num_bins
    )
    mean_rewards = np.mean(rewards, axis=0)
    all_time_avg_reward = np.mean(mean_rewards)

    regret = 195 - all_time_avg_reward

    wandb.log({
        "regret": regret,
        "alpha": config.alpha,
        "epsilon": config.epsilon
    })

    run.finish()

wandb.agent(sweep_id, train, count=50)