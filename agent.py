
# Imports
from env import Beam
import numpy as np
import random
import itertools
from statistics import mean

class Balancer():

    def __init__(self):

        super(Balancer, self).__init__()

        print("Inital Set Up")
        print("================================================================")
        # Initialize Enviroment
        self.env = Beam()

        # Initialize Actions
        self.actions = list(range(-15,20,5))
        self.actions.sort()
        print("Actions:")
        print(self.actions)

        # Set discount rate
        self.GAMMA = 1
        print("Gamma: \n", self.GAMMA)

        # Initalize an arbitrary policy
        print("Initalize Policy:")
        self.policy = {}
        for i in range(len(self.env.states)):
            self.policy[str(self.env.states[i])] = random.choice(self.actions)
        print("Done")

        # Initalize an arbitrary Q-Function
        print("Initalize Q-function:")
        self.qfunc = {}
        for i in range(len(self.env.states)):
            for j in range(len(self.actions)):
                self.qfunc[str(self.env.states[i]+[self.actions[j]])] = 0
        print("Done")

        # Initalize a returns table
        print("Initalize Returns Table:")
        self.returns = {}
        print("Done")

        # Initalize episode counter
        self.n_episodes = 0
        print("================================================================")

        

    def episode(self):
        '''
        This function will run an episode and update the policy and Q-function
        '''

        # Update episode counter
        self.n_episodes += 1

        print("\nEpisode",self.n_episodes, "Summary")
        print("====================================================")

        # Choose a random starting state 
        state = random.choice(self.env.starting_states)

        # Compute action given state
        action = self.policy[str(state)]
        
        # Reset the termination indicator to false
        terminate = False

        # Initalize the history
        history = []
        history.append([state + [action]])

        # Reset returns
        self.returns = {}
        for i in range(len(self.env.states)):
            for j in range(len(self.actions)):
                self.returns[str(self.env.states[i]+[self.actions[j]])] = []
        G = 0
        reward = 0


        # Run episode until termination
        iteration = 0
        while terminate == False and iteration < 10:

            # Update Iteration
            iteration += 1

            # Compute action given state
            action = self.policy[str(state)]

            # Compute Current Returns
            G = G + (self.GAMMA**i)*(reward-1)

            # Store current state action pair pair
            self.returns[str(state + [action])].append(G)

            # Store history
            history.append([state + [action]])
            
            # Take one step
            #with shh(): # shh() suppresses print outs from calls
            state, reward, terminate = self.env.step(state, action)

            if terminate == True:
                # Compute Current Returns
                G = G + (self.GAMMA**i)*(reward-1)

                # Store current state action pair pair
                self.returns[str(history[len(history)-1][0])].append(G)


        #print("History: ")
        #print(history)
        #print("--------------------------------")

        # Find Unique History        
        history.sort()
        unique_history = list(history for history,_ in itertools.groupby(history))
        print("Unique States Visited: ")
        print(unique_history)
        print("--------------------------------")
        
        # Update state-action value for each state visited
        for state_action in unique_history:
            self.qfunc[str(state_action[0])] = mean(self.returns[str(state_action[0])])
            
        # Update the policy for states visited
        print("Policy Updates:")
        for i in range(len(unique_history)):
            state = unique_history[i][0][0:4]

            print('--------------------------------')
            print("State:", state)
            qvalues = []
            for action in self.actions:
                qvalues.append(self.qfunc[str(state+[action])])
            
            print("Old Policy:",self.policy[str(state)])
            # Update policy for state
            self.policy[str(state)] = self.actions[np.argmax(qvalues)]
            print("New Policy", self.policy[str(state)])
        

        print("====================================================\n")


balancer = Balancer()
balancer.episode()