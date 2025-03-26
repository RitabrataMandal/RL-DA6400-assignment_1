import wandb
from sarsa import train_minigrid

sweep_config_grid = {
    "method": "grid",
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "epsilon": {"values": [0.01, 0.05, 0.1, 0.15]},
        "alpha": {"values": [0.1, 0.15, 0.3, 0.5]},
    },
}

sweep_id = wandb.sweep(sweep_config_grid, project="minigrid-sarsa-tuning")

def train():
    run = wandb.init()
    epsilon = wandb.config.epsilon
    alpha = wandb.config.alpha

    mean_reward, _ = train_minigrid(epsilon, alpha)
    # Log results
    wandb.log({"mean_reward": mean_reward, "epsilon": epsilon, "alpha": alpha})
    run.finish()

wandb.agent(sweep_id, train)
