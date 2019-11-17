'''
This will act as the agents mind
'''

# Imports
import numpy as np
import random
import itertools
from statistics import mean

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
        print("====================================================")

        # Choose a random starting state 
        state = random.choice(self.env.starting_states)

        # Compute action given state
        if str(state) in self.policy.keys():
            action = self.policy[str(state)]
        else:
            action = random.choice(self.actions)
            self.policy[str(state)] = action
        
        # Reset the termination indicator to false
        terminate = False

        # Initalize the history
        history = []

        # Reset returns
        state_returns = {}
        state_action_returns = {}
        reward = 0

        # Run episode until termination or limit
        iteration = 0
        while terminate == False and iteration < self.EPISODE_LIMIT:

            # Update Iteration
            iteration += 1

            # Compute action given state
            inital_action = action
            if str(state) in self.policy.keys():
                action = self.policy[str(state)]
            else:
                action = random.choice(self.actions)
                self.policy[str(state)] = action

            
            # Take one step
            #with shh(): # shh() suppresses print outs from calls
            inital_state = state
            state, reward, terminate = self.env.step(state, 
                                                     action, 
                                                     True if iteration == 1 else False, # Episode Reset
                                                     self.EPISODE_LIMIT)

            # Compute Current Returns
            if str(state) in self.policy.keys():
                state_returns[str(inital_state)] += (self.GAMMA**iteration)*(reward)
                state_action_returns[str(inital_state + [inital_action])] += (self.GAMMA**iteration)*(reward)
            else:
                state_returns[str(inital_state)] = (self.GAMMA**iteration)*(reward)
                state_action_returns[str(inital_state + [inital_action])] = (self.GAMMA**iteration)*(reward)

            history.append([inital_state + [inital_action] + [reward] + state + [action]]) # SARSA

        # Display History
        print("\nHistory: ")
        print("State(Inital Position, Current Position, Target Position, Angle), Action, Reward, Next State, Next Action")
        print(history)
        print("--------------------------------")

        # Find Unique History        
        history.sort()
        unique_history = list(history for history,_ in itertools.groupby(history))
        print("Unique States Visited: ")
        print(unique_history)
        print("--------------------------------")
        print("====================================================\n")


        return history

