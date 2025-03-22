# RL-DA6400-Assignment_1
## About
Implementation of SARSA with EpsilonGreedy exploration and Q-learning with softmax exploration policy on three different Gymnasium environments.
1.CartPole-v1
2.MountainCar-v0
3.MiniGrid-Dynamic-Obstacles-5x5-v0

## Environment installation

Create a virtual environment using venv
```bash
python3 -m venv venv --prompt="rl"
```

Activate the virutal environment
```bash
source venv/bin/activate
```

Install Requirements 
```bash
pip install -r requirements.txt
```

## Software packages

To run file on the terminal, run it using 
```bash
python3 main.py
```
To recreate the results run plot_two(_q).py, using 
```bash
python3 plot_two.py
python3 plot_two_q.py
```