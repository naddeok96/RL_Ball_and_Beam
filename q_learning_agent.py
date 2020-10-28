import gym
import itertools
import numpy as np 


class QLearner():

    def __init__(self, env,
                       learning_rate = 0.60,
                       discount_factor = 0.9, 
                       pretrained_q_table = None):
        """Tradition Q Learning Agent where Q-Table is discrete

        Args:
            env (OpenAI type env): enviornment of agent
            learning_rate (float, optional): rate at which to update Q-Table. Defaults to 0.60.
            discount_factor (float, optional): weight to put on future rewards. Defaults to 0.9.
            pretrained_q_table (csv, optional): pretrianed Q-Table. Defaults to None.
        """
        super(QLearner, self).__init__()

        # Load agents enviornment
        self.env = env
        
        # Load Q -Table
        if pretrained_q_table is not None:
            self.q_table = pretrained_q_table
        else:
            self.q_table = np.zeros([np.prod(self.env.observation_space.nvec), # Number of possible observations
                                     self.env.action_space.n]) # Number of possible actions

        # Load Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Get all possible observations
        self._get_state_permutations()

    def _get_state_permutations(self):
        """Takes all possible feature observations and then
            computes all possible observations
        """ 
        # Load all possible feature observations into a list
        state_feature_permutations = []
        for i in range(4):
            state_feature_permutations.append(list(self.env.obs[i]))

        # Compute all possible observations
        self.state_permutations = list(itertools.product(*state_feature_permutations))

    def _get_embedded_state(self, state):
        """Takes in observation and returns unique scalar value

        Args:
            state (tuple): state observation

        Returns:
            int: unique scalar value
        """
        return self.state_permutations.index(state)

    def _get_binned_state(self, raw_state):
        """Takes raw observation and returns binned state

        Args:
            raw_state (tuple): raw observation

        Returns:
            tuple: binned state
        """

        binned_state = []
        for i in range(4):
            binned_state.append(self.env.obs[i][np.abs(self.env.obs[i] - raw_state[i]).argmin()])

        return tuple(binned_state)

    def get_action(self, state):
        """Use Q-Table to find best action given the current state

        Args:
            state (tuple): current binned state

        Returns:
            int: best action to take
        """
        # Find unique scalar value of state
        state = self._get_embedded_state(self._get_binned_state(state))

        # Choose the action with highest value
        return np.argmax(self.q_table[state,:])
    
    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-Table after a step

        Args:
            state (tuple): current raw state
            action (int): action taken
            reward (int): reward recieved
            next_state (tuple): new state after action
        """
        # Bin state then find unique scalar value
        state = self._get_embedded_state(self._get_binned_state(state))
        next_state = self._get_embedded_state(self._get_binned_state(next_state))

        # Calculate new target
        temporal_difference_target = reward + self.discount_factor*np.max(self.q_table[next_state, :])

        # Update Q-Table with Bellmans Equation
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate*(temporal_difference_target - self.q_table[state, action])
        
