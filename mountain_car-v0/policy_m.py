import numpy as np

class EpsilonGreedyPolicy:
    """
        Description: This class implements the epsilon greedy policy.
        Args:
            epsilon  : The exploration rate.
            q_table  : The q-table for the MountainCar-v0 environment.
            env      : The MountainCar-v0 environment.
    """
    def __init__(self, epsilon, q_table, env):
        self.epsilon = epsilon
        self.q_table = q_table
        self.env = env

    def get_action(self, state):
        """
            Description: Returns an action based on the epsilon greedy policy.
            Args:
                state: The current discretized state.
            Returns:
                action: The selected action.
        """
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state[0], state[1]])
        return action

# class SoftmaxPolicy:
#     """
#         Description: This class implements the softmax exploration policy.
#         Args:
#             temperature: The temperature parameter controlling exploration.
#             q_table    : The Q-table for the MountainCar-v0 environment.
#             env        : The MountainCar-v0 environment.
#     """
#     def __init__(self, temperature, q_table, env):
#         self.temperature = temperature
#         self.q_table = q_table
#         self.env = env

#     def get_action(self, state):
#         """
#             Description: Returns an action based on the softmax policy.
#             Args:
#                 state: The current discretized state.
#             Returns:
#                 action: The selected action, chosen probabilistically.
#         """
#         q_values = self.q_table[state[0], state[1]]
#         # Use a numerically stable softmax calculation
#         max_q = np.max(q_values)
#         exp_q = np.exp((q_values - max_q) / self.temperature)
#         probabilities = exp_q / np.sum(exp_q)
#         action = np.random.choice(len(q_values), p=probabilities)
#         return action

class SoftmaxPolicy:
    """
        Description: This class implements the softmax exploration policy.
        Args:
            temperature: The temperature parameter controlling exploration.
            q_table    : The Q-table for the MountainCar-v0 environment.
            env        : The MountainCar-v0 environment.
    """
    def __init__(self, temperature, q_table, env, decay_rate =0.999, min_temp=0.01):
        self.temperature = temperature
        self.q_table = q_table
        self.env = env
        self.decay_rate = decay_rate
        self.min_temp=min_temp

    def get_action(self, state):
        """
            Description: Returns an action based on the softmax policy.
            Args:
                state: The current discretized state.
            Returns:
                action: The selected action, chosen probabilistically.
        """
        q_values = self.q_table[state[0], state[1]]
        # Use a numerically stable softmax calculation
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probabilities)
        self.temperature =min(self.min_temp,self.temperature*self.decay_rate)
        return action