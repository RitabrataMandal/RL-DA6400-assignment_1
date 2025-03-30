# RL-DA6400-Assignment\_1

## About

Implementation of SARSA with Epsilon-Greedy exploration and Q-learning with Softmax exploration policy on three different Gymnasium environments:

- **CartPole-v1**
- **MountainCar-v0**
- **MiniGrid-Dynamic-Obstacles-5x5-v0**

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Steps to Recreate

To set up and run the project, follow these steps:

### 1. Install Required Packages

Run the following command to install all necessary packages:

```bash
pip install -r requirements.txt
```

### 2. Run Experiments for Different Environments

#### 2.1 CartPole-v1

- Navigate to the `cartpole-v1` directory:

```bash
cd cartpole-v1
```

- Run the appropriate script:
  - For SARSA implementation:
  ```bash
  python sarsa.py
  ```
  - For Q-learning implementation:
  ```bash
  python qlearning.py
  ```

#### 2.2 MountainCar-v0

- Navigate to the `mountain_car-v0` directory:

```bash
cd mountain_car-v0
```

- Run the appropriate script:
  - For SARSA implementation:
  ```bash
  python sarsa.py
  ```
  - For Q-learning implementation:
  ```bash
  python q_learning.py
  ```

#### 2.3 MiniGrid-Dynamic-Obstacles-5x5-v0

- Navigate to the `minigrid_world` directory:

```bash
cd minigrid_world
```

- Run the appropriate script:
  - For SARSA implementation:
  ```bash
  python sarsa_epsilon_greedy.py
  ```
  - For Q-learning implementation:
  ```bash
  python q_learning_softmax.py
  ```
