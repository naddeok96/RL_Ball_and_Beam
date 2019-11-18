
'''
This will build an animation of an episode
'''
# Imports
from env import Beam
from agent import Balancer
from silence import shh
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Run One Episode
with shh():
    env = Beam()
    agent = Balancer(env)
    history = agent.episode()

# Summerize Episode
print("Summary")
angles = [0]
ball_x = [history["Step 1"]["Inital State"][1]]
ball_y = 0.15
target_loc = [float([history["Step 1"]["Inital State"][2]][0]),0]
print("Position:", ball_x[0], "Angle:",angles[0])
for i in range(len(history)):
    angle = history["Step " + str(i+1)]["Current State"][3]
    position = history["Step " + str(i+1)]["Current State"][1]
    print("Position:", position, "Angle:",angle)
    angles.append(angle)
    ball_x.append(position)


# Build Animation
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-1, 14), ylim=(-8, 8))
beam = ax.plot([0, 12], [0,0], 'k', linewidth=2, markersize=12)[0]
ball = ax.scatter(ball_x[0], ball_y)
target = ax.scatter(target_loc[0], target_loc[1])
plt.grid()

def animate(i):
    transformation_matrix = np.asarray([[ np.cos(np.deg2rad(angles[i])), -np.sin(np.deg2rad(angles[i]))],
                                        [np.sin(np.deg2rad(angles[i])), np.cos(np.deg2rad(angles[i]))]])
    
    [beam_x_rot, beam_y_rot] = transformation_matrix.dot(np.asarray([12, 0]))
    beam.set_xdata([0, beam_x_rot])
    beam.set_ydata([0, beam_y_rot])

    [ball_x_rot, ball_y_rot] = transformation_matrix.dot(np.asarray([ball_x[i], ball_y]))
    ball.set_offsets(np.hstack((ball_x_rot, ball_y_rot)))

    [target_x_rot, target_y_rot] = transformation_matrix.dot(target_loc)
    target.set_offsets(np.hstack((target_x_rot, target_y_rot)))

anim = FuncAnimation(
    fig, animate, interval= 4*env.TIME_INTERVAL*1000, frames = int(len(angles)))
 
plt.draw()
plt.show()
