import numpy as np

def Qtable(state_space, action_space, bin_size=100):
    bins = np.zeros((state_space.shape[0], bin_size))

    bins[0] = np.linspace(-1.2, 0.6, bin_size)    # position
    bins[1] = np.linspace(-0.07, 0.07, bin_size)    # velocity

    q_table = np.zeros((bin_size, bin_size, action_space.n))

    return q_table, bins


# def discretize_state(state_space, bins):
#     """
#     Discretizes a continuous state into its corresponding bin indices.
#     """
#     state_discrete = np.zeros(state_space.shape)
#     for i in range(state_space.shape[0]):
#         index = np.digitize(state_space[i], bins[i]) - 1
#         state_discrete[i] = min(index, len(bins[i]) - 1)
#     return state_discrete.astype(np.int32)


def discretize_state(state_space, bins):
    state_discrete = np.zeros(state_space.shape,dtype=int)

    for i in range(state_space.shape[0]):
        index = np.digitize(state_space[i],bins[i])-1
        state_discrete[i]=np.clip(index, 0,len(bins[i])-1)


    return tuple(state_discrete)