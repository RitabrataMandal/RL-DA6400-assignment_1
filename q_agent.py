import numpy as np
from q_table import discretize_state
from policy import EpsilonGreedyPolicy, SoftmaxPolicy

class QLearner:
    
    def __init__(self, alpha, gamma,temperature, q_table, bins, env, seed):
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.q_table = q_table
        self.env = env
        self.bins = bins
        self.seed = seed
        self.policy = SoftmaxPolicy(self.temperature, self.q_table, self.env)

    def compute_td_error(self, state, action, next_state, reward):
        """
            Description: Computes the TD error.
            Args:
                state      : The current state.
                action     : The current action.
                next_state : The next state.
                reward     : The reward.
            Returns:
                td_error   : The computed TD error.
        """
        return reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], next_state[2], next_state[3]]) - \
               self.q_table[state[0], state[1], state[2], state[3], action]

    def update_q_table(self, state, action, td_error):
        """
            Description: Updates the Q-table.
            Args:
                state    : The current state.
                action   : The current action.
                td_error : The TD error.
        """
        self.q_table[state[0], state[1], state[2], state[3], action] += self.alpha * td_error

    def learn(self, num_episodes, num_steps):
        """
            Description: Implements the Q-Learning procedure.
            Args:
                num_episodes: Number of episodes.
                num_steps   : Number of steps per episode.
            Returns:
                reward_list : List of total rewards per episode.
        """
        reward_list = []
        for episode in range(num_episodes):
            # Reset the environment with the given seed for reproducibility
            state, _ = self.env.reset(seed=self.seed)
            state_discrete = discretize_state(state, self.bins)
            total_reward = 0
            for step in range(num_steps):
                action = self.policy.get_action(state_discrete)
                next_state, reward, done = self.env.step(action)[:3]
                next_state_discrete = discretize_state(next_state, self.bins)
                td_error = self.compute_td_error(state_discrete, action, next_state_discrete, reward)
                self.update_q_table(state_discrete, action, td_error)
                state = next_state
                state_discrete = next_state_discrete
                total_reward += reward
                if done or step == num_steps - 1:
                    print("Episode: {}/{}, Total Reward: {}".format(episode + 1, num_episodes, total_reward))
                    reward_list.append(total_reward)
                    break
        return reward_list
