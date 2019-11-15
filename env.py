
'''
This will initalize the possible states and simulate the state transitions given an action
'''

# Imports
import numpy as np

class Beam():

    def __init__(self):

        super(Beam, self).__init__()

        # Initialize States [p0,p,t,ang]
        self.states = []
        for i in range(13): # Inital ball location [p0]
            for j in range(13): # Current ball location [p]
                for k in range(13): # Target ball location [t]
                    for m in range(17): # Beam angle [ang]
                        if m == 16:
                            ang = 0
                        else:
                            ang = (m*6) - 45
                        self.states.append([i,j,k,ang])
        print("State Space Size:")
        print(np.shape(self.states))    

        # Initialize Starting States
        self.starting_states = []
        for i in range(13): # Current ball location [p]
            for j in range(13): # Target ball location [t]
                self.starting_states.append([i,i,j,0])
        print("Starting State Space Size:")
        print(np.shape(self.starting_states))

        # Initialize Terminal States
        self.terminal_states = []
        for i in range(13): # Target ball location [t]
            self.terminal_states.append([i,i,i,0])
        print("Terminal State Space Size:")
        print(np.shape(self.terminal_states))

        # Environment Dynamics
        self.ACC_GRAV = 386.09 # [in/s^2]
        self.TIME_INTERVAL = 0.25 # [s]
        self.terminate = False

    def step(self, state, action):

        
        # Take a time step
        intial_position = state[1]

        inital_velocity = (state[1]-state[0])/self.TIME_INTERVAL

        angle = self.limit_angle(state[3] + action)

        acceleration = -self.ACC_GRAV*np.sin(np.deg2rad(angle))

        velocity = round(inital_velocity + acceleration*self.TIME_INTERVAL)

        position = self.bin_position(intial_position + (inital_velocity*self.TIME_INTERVAL) + (0.5*acceleration*(self.TIME_INTERVAL**2)))

        next_state = [intial_position, position, state[2], angle]

        reward = 0
        if next_state in self.terminal_states:
            reward = 1

        if next_state not in self.states:
            reward = -1
            terminate = TRUE
        
        # Display Summary
        print("\nStep Summary")
        print("Reward: ",reward)
        print("Episode Terminated: ",terminate)
        print("----------------------------------------------------")
        print("Inital Position | Inital Velocity | Initial Angle  |")
        print("\t",intial_position, "\t|\t", inital_velocity, "\t  |\t", state[3], "\t   |")
        print(" Final Position | Final Velocity  | Final Angle    |")
        print("\t",position, "\t|\t", velocity, "\t  |\t", angle, "\t   |")
        print("----------------------------------------------------")

        return next_state, reward, terminate

    def bin_position(self, position):
        '''
        Bin the position
        '''

        bins = np.linspace(0.5, 12.5, 13)
        digitized = np.digitize(position, bins)

        return digitized

    def limit_angle(self, angle):
        '''
        Keep the angle within limits
        '''

        if angle > 45:
            angle = 45
        if angle < -45:
            angle = -45
        return angle

