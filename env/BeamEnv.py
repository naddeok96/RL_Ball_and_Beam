# Imports
import gym
import numpy as np
import random
import matplotlib as mpl
from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import math
import os


class BeamEnv(gym.Env):

    def __init__(self, obs_low_bounds  = np.array([  0,   0,  1.18e10, -45]),
                       obs_high_bounds = np.array([ 12,  12, -1.18e10,  45]), 
                       obs_bin_sizes   = np.array([0.5, 0.5,        6,   5]),
                       TIME_STEP   = 0.1):
        """Environment for a ball and beam system where agent has control of tilt.

        Args:
            obs_low_bounds (list, optional): [target location(in), ball location(in), 
                                              ball velocity(in/s), beam angle(deg)]. Defaults to [ 0,  0, "TBD", -45].
            obs_high_bounds (list, optional): As above so below. Defaults to [12, 12, "TBD",  45].
        """
        super(BeamEnv, self).__init__()

        # Hyperparameters
        self.ACC_GRAV    = 386.22  # [in/s2]
        self.MOTOR_SPEED = 46.875  # 1.28[sec/60deg] converted to [deg/s]
        self.TIME_STEP   = TIME_STEP # [s]

        # Declare bounds of observations
        self.obs_low_bounds  = obs_low_bounds
        self.obs_high_bounds = obs_high_bounds
        self._set_velocity_bounds()

        # Declare bin sizes of observations
        self.obs_bin_sizes   = obs_bin_sizes
        
        # Bin observations
        self.binned_obs = []
        self.obs_sizes = []
        for i in range(4):
            self.binned_obs.append(np.sort(
                            np.append(
                            np.arange(self.obs_low_bounds[i], self.obs_high_bounds[i] + self.obs_bin_sizes[i], self.obs_bin_sizes[i]), 
                            0)))

            self.obs_sizes.append(len(self.binned_obs[i]))
        
        # Declare observation space
        self.observation_space = gym.spaces.MultiDiscrete(self.obs_sizes)

        # Action Space
        # increase, decrease or keep current angle
        self.action_space = gym.spaces.Discrete(3)

        # Increments each time after receiving reward
        self.step_counter = 0

        # Set number of export frames
        self.sample_freq = 10
        
        # Set the reward_range if you want a narrower range
        # Defaults to [-inf,+inf]
        self.reward_range = (-1, 1)
   
    
    def _init_plt(self):
        """Visualizes a plot using Matplotlib.   
        """

        plt.style.use('dark_background')

        if self.fig: return
        plt.close()

        # Create a figure on the screen
        fig = plt.figure(figsize = (5, 5))
        self.fig = fig
        axes = plt.axes(xlim = (-5, 20), ylim = (0, 30))

        # Set the aspect of the axis scaling
        axes.set_aspect('equal')
        self.axes = axes

        #Plot a circle to illustrate the ball
        circle = plt.Circle((self.ball_location, 15), radius = 0.75, ec ='#c77dff', fc ='#c77dff')
        self.circle = circle
        # Return the figure element patch
        axes.add_patch(circle)
        
        #Plot a rectangle to illustrate the beam
        rect = plt.Rectangle((-3, 14.97), width = 20, height = 0.25, ec = '#adb5bd', fc = '#adb5bd')
        self.rect = rect
        axes.add_patch(rect)

        # Set x coord to the target location
        x = self.target_location
        a = 1.5
        
        # Calculate target coordinates [x, b], [x - a, b], [x + a, b]
        pivot = plt.Polygon([[x, 15], [x - a, 12], [x + a, 12]], fc = '#adb5bd')
        axes.add_patch(pivot)

        plt.show(block = False)
        plt.pause(0.10)

    def _render(self, obs):
        """Renders a 2D environment.
        Args:
            obs (int): The number of possible discrete actions.
        """

        # Draw the figure
        self.fig.canvas.draw()
        
        # Set the initial horizontal position of the ball
        ball_loc = obs[1]
       
        # Calculate the new y position
        new_y = math.tan(-math.radians(obs[3])) * (ball_loc - self.target_location)

        # Set the center of the circle (x, y, radius)
        self.circle.center = (ball_loc, 15 + new_y + 1)
        
        # Convert data coordinates to display coordinates
        ts = self.axes.transData
        xy = (self.target_location, 15)

        # Set data coordinate system
        # Transform ensures the plot will be correct
        coords = ts.transform(xy)
        
        # Image rotation
        tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], -obs[3])
        
        # Transform to display coordinates then rotate
        t = ts + tr
        # t is the Transform
        self.rect.set_transform(t)
        plt.pause(0.10)


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
        vel_max = round(np.sqrt(2*a_max*distance_max))

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
        self.fig = None
        self.target_location = target_location if target_location is not None else random.choice(self.binned_obs[0])

        # Set ball location
        self.ball_location = ball_location if ball_location is not None else random.choice(self.binned_obs[1])

        # Set Intial Velocity and Angle to Zero
        self.ball_velocity = 0.0 # [in/s]
        self.beam_angle    = 0.0 # [deg]

        obs = self._next_observation()
        return obs

    def _next_observation(self):
        """Determines what will happen in the next time step

        Returns:
            tuple: observation of (target location, ball location, ball velocity, beam angle)
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
        return (self.target_location, self.ball_location, self.ball_velocity, self.beam_angle)

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

    def export_frames(self):
        """ Saves samples of the environment frames.
        """
        path = 'samples/'
        os.path.abspath(__file__)
        my_file = '.png'
        plt.savefig(path + str(self.step_counter) + my_file)

    def step(self, state, action, render=False):
        """Take action then collect reward and get new observation

        Args:
            state (tuple): current state
            action (int): increase, decrease or keep current angle
            render (bool, optional): If True beam is displayed. Defaults to False

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Take the action
        self._take_action(action)

        # Determine Success
        ball_is_on_target = (round(abs((self.target_location - self.ball_location)),3) == 0)
        ball_is_stopped   = (round(self.ball_velocity, 3) == 0)
        beam_is_level     = (round(self.beam_angle, 3) == 0)

        done = True if ball_is_on_target & ball_is_stopped & beam_is_level else False

        # Find Reward
        reward = 1 if done else -1

        # Find Next Observation
        obs = self._next_observation()
        
        # Terminate rendering 
        if render:
            self.step_counter += 1
            self._init_plt()
            self._render(obs)
            if self.step_counter % self.sample_freq == 0:
                self.export_frames()
        return obs, reward, done