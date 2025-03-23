import numpy as np
from q_table import discretize_state
from policy import SoftmaxPolicy

class QLearner:
    
    def __init__(self, alpha, gamma, temperature, q_table, bins, env, seed):
        self.alpha = alpha  
        self.gamma = gamma  
        self.temperature = temperature  
        self.q_table = q_table  
        self.bins = bins  
        self.env = env  
        self.seed = seed 
        self.policy = SoftmaxPolicy(self.temperature, self.q_table, self.env)  
    
    def compute_td_error(self, state, action, next_state, reward):
        """
            Computes the TD error for Q-learning.
        """
        best_next_action_value = np.max(self.q_table[next_state[0], next_state[1], :])
        return reward + self.gamma * best_next_action_value - self.q_table[state[0], state[1], action]

    def update_q_table(self, state, action, td_error):
        """
            Updates the Q-table using the TD error.
        """
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    def learn(self, num_episodes, num_steps):
        """
        Implements the Q-Learning algorithm for MountainCar-v0.
        """
        reward_list = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed)
            state_discrete = discretize_state(state, self.bins)
            total_reward = 0
            
            for step in range(num_steps):
                action = self.policy.get_action(state_discrete)
                # Unpack all returned values from env.step
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_discrete = discretize_state(next_state, self.bins)
                
                # Compute TD error and update Q-table
                td_error = self.compute_td_error(state_discrete, action, next_state_discrete, reward)
                self.update_q_table(state_discrete, action, td_error)
                
                state_discrete = next_state_discrete
                total_reward += reward
                
                if done or truncated:
                    break
                    
            print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
            reward_list.append(total_reward)
        
        return reward_list

