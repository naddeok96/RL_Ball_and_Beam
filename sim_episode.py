
'''
This will build an animation of an episode
'''
from env import Beam
from agent import Balancer
from silence import shh
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Run One Episode

env = Beam()
agent = Balancer(env)

history = agent.episode()

print("Summary")
angles = []
ball_x = [history[0][1]]
ball_y = [0]
beam_x = [[0,12]]
beam_y = [[0,0]]
for step in history:
    
    angle = step[9]
    position = step[7]
    print("Position:", position, "Angle:",angle)
    angles.append(angle)
    ball_x.append(position*np.cos(np.deg2rad(angle)))
    ball_y.append(position*np.sin(np.deg2rad(angle)))

    beam_x.append([0,12*np.cos(np.deg2rad(angle))])
    beam_y.append([0,12*np.sin(np.deg2rad(angle))])


fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-1, 14), ylim=(-6, 6))
beam = ax.plot(beam_x[0], beam_y[0], 'k', linewidth=2, markersize=12)[0]
ball = ax.scatter(ball_x[0], ball_y[0])
plt.grid()

def animate(i):
    beam.set_xdata(beam_x[i])
    beam.set_ydata(beam_y[i])
    #print("Ball Location:",np.hstack((ball_x[i], ball_y[i])) )
    ball.set_offsets(np.hstack((ball_x[i], ball_y[i])))
    



anim = FuncAnimation(
    fig, animate, interval= 4*env.TIME_INTERVAL*1000, frames = int(len(beam_x)))
 
plt.draw()
plt.show()
