import wandb
from q_learning import train_qlearning_softmax

sweep_config = {
    "method": "grid",
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "temperature": {"values": [0.5, 1, 1.5]},
        "alpha": {"values": [0.1, 0.2, 0.3]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning-softmax-tuning")

def train():
    run = wandb.init()
    temperature = wandb.config.temperature
    alpha = wandb.config.alpha

    mean_reward, _ = train_qlearning_softmax(temperature, alpha)
    wandb.log({"mean_reward": mean_reward, "temperature": temperature, "alpha": alpha})
    run.finish()

wandb.agent(sweep_id, train)
