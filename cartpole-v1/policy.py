import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon, q_table, env):
        self.epsilon = epsilon
        self.q_table = q_table
        self.env = env

    def get_action(self, state):
       
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state[0], state[1], state[2], state[3]])
        return action

class SoftmaxPolicy:
    def __init__(self, temperature, q_table, env):
        self.temperature = temperature
        self.q_table = q_table
        self.env = env

    def get_action(self, state):
        q_values = self.q_table[state[0], state[1], state[2], state[3]]
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probabilities)
        return action

