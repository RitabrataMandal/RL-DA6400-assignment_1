import wandb
from q_learning_softmax import train_q_learning  # Import the Q-learning function

# Initialize wandb project
# wandb.init(project="qlearning_mountain_car-v0tuning")

# Hyperparameter tuning
sweep_config = {
    "method": "grid",  # Can use "random" or "bayes" for optimization
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "temperature": {"values": [0.1, 0.5, 0.8, 1.0, 1.5]},  
        "alpha": {"values": [0.1, 0.3, 0.5, 0.7, 0.9]},  
    },
}

# Create a new sweep
sweep_id = wandb.sweep(sweep_config, project="qlearning_mountain_car-v0tuning")

def train():
    with wandb.init(project="qlearning_mountain_car-v0tuning") as run:
        temperature = wandb.config.temperature
        alpha = wandb.config.alpha

        # Run Q-learning
        mean_reward = train_q_learning(temperature, alpha)

        # Log results
        wandb.log({"mean_reward": mean_reward, "temperature": temperature, "alpha": alpha})



# Run the sweep
wandb.agent(sweep_id, train)
