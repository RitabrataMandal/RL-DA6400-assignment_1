import numpy as np

def Qtable(state_space, action_space, bin_size=100):
    
    bins = np.zeros((state_space.shape[0], bin_size))

    bins[0] = np.linspace(-4.8, 4.8, bin_size)
    bins[1] = np.linspace(-4, 4, bin_size)
    bins[2] = np.linspace(-0.42, 0.42, bin_size)
    bins[3] = np.linspace(-4, 4, bin_size)

    q_table = np.zeros((bin_size, bin_size, bin_size, bin_size, action_space.n))

    return q_table, bins


def discretize_state(state_space, bins):
    
    state_discrete = np.zeros(state_space.shape)

    for i in range(state_space.shape[0]):
        state_discrete[i] = np.digitize(state_space[i], bins[i])

    return state_discrete.astype(np.int32)