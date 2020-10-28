import gym
import itertools
import numpy as np 


class QLearner():

    def __init__(self, env,
                       learning_rate = 0.60,
                       discount_factor = 0.9, 
                       pretrained_q_table = None):
        super(QLearner, self).__init__()

        self.env = env
        
        if pretrained_q_table is not None:
            self.q_table = pretrained_q_table
        else:
            self.q_table = np.zeros([np.prod(self.env.observation_space.nvec), # Number of possible observations
                                     self.env.action_space.n]) # Number of possible actions

        self._get_state_permutations()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def _get_state_permutations(self):
        state_feature_permutations = []
        for i in range(4):
            state_feature_permutations.append(list(self.env.obs[i]))

        self.state_permutations = list(itertools.product(*state_feature_permutations))

    def _get_embedded_state(self, state):

        return self.state_permutations.index(state)

    def _get_binned_state(self, raw_state):

        binned_state = []
        for i in range(4):
            binned_state.append(self.env.obs[i][np.abs(self.env.obs[i] - raw_state[i]).argmin()])

        return tuple(binned_state)

    def get_action(self, state):
        state = self._get_embedded_state(self._get_binned_state(state))
        return np.argmax(self.q_table[state,:])
    
    def update_q_table(self, state, action, reward, next_state):
        state = self._get_embedded_state(self._get_binned_state(state))
        next_state = self._get_embedded_state(self._get_binned_state(next_state))

        # Calculate new target
        temporal_difference_target = reward + self.discount_factor*np.max(self.q_table[next_state, :])

        # Update Q-Table with Bellmans Equation
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate*(temporal_difference_target - self.q_table[state, action])
        
