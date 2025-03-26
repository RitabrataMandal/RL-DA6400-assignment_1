import wandb
from sarsa_epsilon import train_sarsa  # Import the SARSA training function

# wandb.init(project="sarsa_mountain_car-v0tuning")

# Define the sweep configuration (you can adjust the parameter values as needed)
sweep_config = {
    "method": "grid",  # Or "random", "bayes"
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "epsilon": {"values": [0.01, 0.05, 0.1]},  
        "alpha": {"values": [0.1, 0.3, 0.5]},  
    },
}

# Create a new sweep
sweep_id = wandb.sweep(sweep_config, project="sarsa-mountaincar-v0tuning")

def train():
    run = wandb.init()  # Initialize wandb run
    # Get hyperparameters from wandb config
    epsilon = wandb.config.epsilon
    alpha = wandb.config.alpha

    # Run the SARSA training with the given hyperparameters
    mean_reward = train_sarsa(epsilon, alpha)

    # Log the results
    wandb.log({"mean_reward": mean_reward, "epsilon": epsilon, "alpha": alpha})
    run.finish()

# Start the wandb agent for the sweep
wandb.agent(sweep_id, train)
