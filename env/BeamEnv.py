# Imports
import gym
import numpy as np
import random

class BeamEnv(gym.Env):

    def __init__(self, obs_low_bounds  = np.array([ 0,  0, "TBD", -45]),
                       obs_high_bounds = np.array([12, 12, "TBD",  45])):
        """Environment for a ball and beam system where agent has control of tilt.

        Args:
            obs_low_bounds (list, optional): [target location(in), ball location(in), 
                                              ball velocity(in/s), beam angle(deg)]. Defaults to [ 0,  0, "TBD", -45].
            obs_high_bounds (list, optional): As above so below. Defaults to [12, 12, "TBD",  45].
        """
        super(BeamEnv, self).__init__()

        # Hyperparameters
        self.ACC_GRAV    = 386.22  # [in/s]
        self.MOTOR_SPEED = 46.875  # 1.28[sec/60deg] converted to [deg/s]
        self.TIME_STEP   = 0.1     # [s]

        # Observation Space 
        #  _bounds = []
        self.obs_low_bounds  = obs_low_bounds
        self.obs_high_bounds = obs_high_bounds
        self._determine_max_velocity()
        self.observation_space = gym.spaces.Box(low = self.obs_low_bounds,
                                                high = self.obs_high_bounds,
                                                dtype = np.float32)

        # Action Space
        # increase, decrease or keep current angle
        self.action_space = gym.spaces.Descrete(3)

        # Reward Range
        self.reward_range = (-1, 1)

    def _set_velocity_bounds(self):
        """Use Inclined Plane and Kinematics Formulas 
            to determine min/max velocities and set the obs_low/high_bounds
        """
        # Max Distance
        distance_max = self.obs_high_bounds[1]

        # Max Angle
        ang_max = self.obs_high_bounds[3]

        # Max acceletation (Inclined Plane Formula)
        a_max = self.ACC_GRAV * np.sin(np.deg2rad(ang_max))

        # Max Velocity (vf^2 = v0^2 + 2ad)
        vel_max = np.sqrt(2*a_max*distance_max)

        # Set Bounds
        self.obs_low_bounds[2]  = -vel_max
        self.obs_high_bounds[2] =  vel_max

    def reset(self, target_location = None,
                    ball_location = None):
        """Reset the environment so the ball is not moving, there is no angle,


        Args:
            target_location (float, optional): Desired location of ball. Defaults to random.
            ball_location (float, optional): Current location of ball. Defaults to random.

        Returns:
            list: observation of (target location, ball location, ball velocity, beam angle)
        """
        
        # Set target location
        if target_location is not None:
            self.target_location = target_location
        else:
            possible_targets = list(range(self.obs_low_bounds[0], self.obs_high_bounds[0]))
            self.target_location = random.choice(possible_targets)

        # Set ball location
        if ball_location is not None:
            self.ball_location = ball_location
        else:
            possible_ball_locations = list(range(self.obs_low_bounds[1], self.obs_high_bounds[1]))
            self.ball_location = random.choice(possible_ball_locations)

        # Set Intial Velocity and Angle to Zero
        self.ball_velocity = 0 # [in/s]
        self.beam_angle    = 0 # [deg]

        return self._next_observation()

    def _next_observation(self):
        """Determines what will happen in the next time step

        Returns:
            list: observation of (target location, ball location, ball velocity, beam angle)
        """
        # Calculate Acceleration (Inclined Plane Equation)
        ball_acceleration = self.ACC_GRAV * np.sin(np.deg2rad(self.beam_angle))

        # Calculate Next Position (x = x0 + v0t + 0.5at^2)
        self.ball_location = self.ball_location + self.ball_velocity * self.TIME_STEP + 0.5 * ball_acceleration * self.TIME_STEP**2

        # Calculate New Velocity (v = v0 + at)
        self.ball_velocity = self.ball_velocity + ball_acceleration * self.TIME_STEP

        # Clip Ball Location
        self.ball_location = max(min(self.ball_location, 
                                     self.obs_high_bounds[1]),
                                     self.obs_low_bounds[1])

        # Clip Ball Velocity
        self.ball_velocity = max(min(self.ball_velocity, 
                                     self.obs_high_bounds[2]),
                                     self.obs_low_bounds[2])

        # Return Observation
        return [self.target_location, self.ball_location, self.ball_velocity, self.beam_angle]    

    def _take_action(self,action):
        """Determines change in angle due to action

        Args:
            action (int): increase, decrease or keep current angle
        """
        # Change action to signs by subtracting by 1 ie (0,1,2) --> (-1,0,1)
        action -= 1

        # Change the angle by unit step
        self.beam_angle = self.beam_angle + action * self.MOTOR_SPEED * self.TIME_STEP

        # Clip
        self.beam_angle = max(min(self.beam_angle, 
                                     self.obs_high_bounds[3]),
                                     self.obs_low_bounds[3])

    def step(self, action):
        """Take action, collect reward and get new observation

        Args:
            action (int): increase, decrease or keep current angle

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Take the action
        self._take_action(action)

        # Determine Success
        if (round(abs((self.target_location - self.ball_location)),3) == 0) & (round(self.ball_velocity, 3) == 0) & (round(self.beam_angle, 3) == 0):
            done = True
        else:
            done = False

        # Find Reward
        reward = 1 if done else -1

        # Find Next Observation
        obs = self._next_observation()

        # Return what happened
        return obs, reward, done, {}



         
        



