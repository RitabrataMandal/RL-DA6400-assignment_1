import wandb
import yaml
from q_learning import train_qlearning_softmax

# Load sweep configuration from the YAML file
with open("sweep_config.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning-softmax-tuning")

def train():
    run = wandb.init()
    
    temperature = wandb.config.temperature
    alpha = wandb.config.alpha

    mean_reward, _ = train_qlearning_softmax(temperature, alpha)
    
    regret = 1 - mean_reward
    
    wandb.log({
        "regret": regret,
        "temperature": temperature,
        "alpha": alpha
    })
    
    run.finish()

wandb.agent(sweep_id, train, count=50)
