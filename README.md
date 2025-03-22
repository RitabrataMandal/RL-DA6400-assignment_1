# RL-DA6400-Assignment_1

## About
Implementation of SARSA with EpsilonGreedy exploration and Q-learning with softmax exploration policy on three different Gymnasium environments:

- **CartPole-v1**  
- **MountainCar-v0**  
- **MiniGrid-Dynamic-Obstacles-5x5-v0**  

## Environment Installation

Create a virtual environment using venv:
```bash
python3 -m venv venv --prompt="rl"
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Software Packages

To run the file on the terminal, use:
```bash
python3 main.py
```

To recreate the results, run `plot_two(_q).py` using:
```bash
python3 plot_two.py
python3 plot_two_q.py
