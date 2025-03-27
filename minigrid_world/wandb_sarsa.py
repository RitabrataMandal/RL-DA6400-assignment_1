import wandb
import yaml
from sarsa import train_minigrid



with open("sweep_sarsa.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)



sweep_id = wandb.sweep(sweep_config, project="minigrid-sarsa-tuning")

def train():
    run = wandb.init()
    epsilon = wandb.config.epsilon
    alpha = wandb.config.alpha

    mean_reward, _ = train_minigrid(epsilon, alpha)
    regret = 1-mean_reward
    # Log results
    wandb.log({"regret": regret, "epsilon": epsilon, "alpha": alpha})
    run.finish()

wandb.agent(sweep_id, train, count=50)
