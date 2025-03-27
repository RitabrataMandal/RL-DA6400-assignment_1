import wandb
import yaml
from sarsa_epsilon import train_sarsa  


with open("sweep_sarsa.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="sarsa-mountaincar-v0tuning")

def train():
    run = wandb.init()  
    epsilon = wandb.config.epsilon
    alpha = wandb.config.alpha

    mean_reward = train_sarsa(epsilon, alpha)
    regret = -200-mean_reward

    wandb.log({"regret": regret, "epsilon": epsilon, "alpha": alpha})
    run.finish()

wandb.agent(sweep_id, train, count=50)
