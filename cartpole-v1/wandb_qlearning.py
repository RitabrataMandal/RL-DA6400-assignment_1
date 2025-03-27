import wandb
import yaml
import numpy as np
from q_runner import run_qlearning  

# Load sweep configuration from the YAML file
with open("sweep_qlearning.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="qlearning-cartpole")

def train():
    run = wandb.init()
    config = wandb.config
    
    gamma = 0.99
    env_name = "MountainCar-v0"
    seeds = [100, 200, 300, 400, 500]
    num_episodes = 1000
    num_steps = 500
    num_bins = 20

    rewards = run_qlearning(
        env_name,
        config.alpha,
        gamma,
        config.temperature,
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
        "temperature": config.temperature
    })
    
    run.finish()

wandb.agent(sweep_id, train, count=50)
