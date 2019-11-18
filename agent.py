'''
This will act as the agents mind
'''

# Imports
import numpy as np
import random
import itertools
from statistics import mean
from silence import shh

class Balancer():

    def __init__(self, env,
                       actions = list(range(-15,20,5)),
                       EPISODE_LIMIT = 10,
                       GAMMA = 1):

        super(Balancer, self).__init__()

        print("Agent Set Up")
        print("================================================================")
        # Initialize Enviroment
        self.env = env

        # Initialize Actions
        self.actions = actions
        self.actions.sort()
        print("Actions:")
        print(self.actions)

        # Set discount rate
        self.GAMMA = GAMMA
        print("Gamma: \n", self.GAMMA)

        # Set episode limit
        self.EPISODE_LIMIT = EPISODE_LIMIT
        print("Episode Limit: \n", self.EPISODE_LIMIT)

        # Initalize an arbitrary Policy, Value and Q-Function
        self.policy = {}
        self.value = {}
        self.qfunc = {}
        

        # Initalize episode counter
        self.n_episodes = 0
        print("================================================================")
        

    def episode(self):
        '''
        This function will run an episode
        '''

        # Update episode counter
        self.n_episodes += 1

        print("\nEpisode",self.n_episodes, "Summary")
        print("========================================================================================================================")

        # Choose a random starting state 
        state = random.choice(self.env.starting_states)

        # Compute action given state
        action = self.getPolicy(self.policy, str(state))
        
        # Reset everything
        terminate = False
        history = {}
        state_returns = {}
        state_action_returns = {}
        reward = 0
        iteration = 0
        while  iteration < self.EPISODE_LIMIT: #terminate == False and

            # Update Iteration
            iteration += 1

            # Take one step
            with shh(): # shh() suppresses print outs from calls
                inital_state = state
                state, reward, terminate = self.env.step(state, 
                                                        action, 
                                                        True if iteration == 1 else False, # Episode Reset
                                                        self.EPISODE_LIMIT)

            # Compute action given state
            inital_action = action
            action = self.getPolicy(self.policy, str(state))

            # Compute Current Returns
            self.setReturn(state_returns, str(state), (self.GAMMA**iteration)*reward)
            self.setReturn(state_action_returns, str(state + [action]), (self.GAMMA**iteration)*reward)

            history["Step " + str(iteration)] = {'Inital State': inital_state,
                                                 'Inital Action': inital_action,
                                                 'Reward': reward,
                                                 'Current State': state,
                                                 'Current Action': action}
            

        # Display History
        print("\nHistory: (states are of [p0,p,t,angle])")

        for i in range(iteration):
            print("Step " + str(i+1))
            print(history["Step " + str(i+1)])
        print("========================================================================================================================")

        return history

    def setReturn(self,return_func, key,value):

        if key in return_func.keys():
            return_func[key] += value
        else:
            return_func[key] = value

    def getPolicy(self,policy,key):

        if key in policy.keys():
            action = policy[key]
        else:
            action = random.choice(self.actions)
            policy[key] = action

        return action

