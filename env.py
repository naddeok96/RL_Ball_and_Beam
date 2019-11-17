
'''
This will initalize the possible states and simulate the state transitions given an action
'''

# Imports
import numpy as np

class Beam():

    def __init__(self, position_scale = 1,
                       angle_limit = 45):

        super(Beam, self).__init__()

        print("\nEnvironment Setup: ")
        print("================================================================")
        # Initalize Position
        self.position = 0
        self.intial_position = 0
        self.position_scale = position_scale
        set_size = 1 + 12/position_scale
        self.positions = np.linspace(0, 12, set_size)
        print("Positions: \n", self.positions)

        # Initialize Velocity
        self.inital_velocity = 0
        self.velocity = 0

        # Initalize Angle Limits
        self.angle_limit = angle_limit
        print("Angle Limit: \n", self.angle_limit, "[deg]")

        # Initialize States [p0,p,t,ang]
        self.states = []
        for i in range(13): # Inital ball location [p0]
            for position in self.positions: # Current ball location [p]
                for k in range(13): # Target ball location [t]
                    for m in range(19): # Beam angle [ang]
                        if m == 19:
                            ang = 0
                        else:
                            ang = (m*5) - 45
                        self.states.append([i,position,k,ang])
        print("State Set: \n[Inital Position, Current Position, Target Position, Beam Angle]")
        print("State Space Size: \n", np.shape(self.states)[0])

        # Initialize Starting States
        self.starting_states = []
        for i in range(13): # Current ball location [p]
            for j in range(13): # Target ball location [t]
                self.starting_states.append([i,i,j,0])
        print("Starting State Space Size: \n", np.shape(self.starting_states)[0])

        # Initialize Terminal States
        self.terminal_states = []
        for i in range(13): # Target ball location [t]
            self.terminal_states.append([i,i,i,0])
        print("Terminal State Space Size: \n", np.shape(self.terminal_states)[0])

        # Environment Dynamics
        self.ACC_GRAV = 386.09 # [in/s^2]
        self.TIME_INTERVAL = 0.25 # [s]
        print("Time Interval: \n", self.TIME_INTERVAL,"[s]")
        print("================================================================\n")
        

    def step(self, state, action, episode_reset, episode_limit):

        terminate = False

        if episode_reset == True:
            self.intial_position = state[1]
            self.position = state[1]
            self.inital_velocity = 0
            self.velocity = 0
        else:
            self.intial_position = self.position
            self.inital_velocity = self.velocity

        angle = self.limit_angle(state[3] + action)

        acceleration = -self.ACC_GRAV*np.sin(np.deg2rad(angle))

        self.velocity = self.inital_velocity + acceleration*self.TIME_INTERVAL

        self.position = self.intial_position + (self.inital_velocity*self.TIME_INTERVAL) + (0.5*acceleration*(self.TIME_INTERVAL**2))

        next_state = [state[0], int(self.bin_position(self.position)), state[2], int(angle)]

        reward = -1
        if next_state in self.terminal_states:
            reward = episode_limit

        if next_state not in self.states:
            reward = -episode_limit
            terminate = True
        
        # Display Summary
        print("\nStep Summary")
        print("Action: ",action)
        print("Reward: ",reward)
        print("Episode Terminated: ",terminate)
        print("----------------------------------------------------")
        print("Inital Position | Inital Velocity | Initial Angle  |")
        print("\t",self.bin_position(self.intial_position), "\t|\t", round(self.inital_velocity), "\t  |\t", state[3], "\t   |")
        print(" Final Position | Final Velocity  | Final Angle    |")
        print("\t",self.bin_position(self.position), "\t|\t", round(self.velocity), "\t  |\t", angle, "\t   |")
        print("----------------------------------------------------")

        return next_state, reward, terminate

    def bin_position(self, position):
        '''
        Bin the position
        '''
        return min([-1,13] + list(self.positions), key=lambda x:abs(x-position))  

    def limit_angle(self, angle):
        '''
        Keep the angle within limits
        '''
        if angle > self.angle_limit:
            angle = self.angle_limit
        if angle < -self.angle_limit:
            angle = -self.angle_limit
        return angle

