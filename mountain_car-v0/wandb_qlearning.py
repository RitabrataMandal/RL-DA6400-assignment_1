import wandb
import yaml
from q_learning_softmax import train_q_learning 

with open("sweep_qlearning.yaml","r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="qlearning_mountain_car-v0tuning")

def train():
        run =wandb.init()
        temperature = wandb.config.temperature
        alpha = wandb.config.alpha

        mean_reward = train_q_learning(temperature, alpha)
        regret = -200-mean_reward

        wandb.log({"regret": regret, "temperature": temperature, "alpha": alpha})
        run.finish()

wandb.agent(sweep_id, train, count=50)
