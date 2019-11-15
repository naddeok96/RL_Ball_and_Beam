# RL Ball and Beam
The goal of this project is to utilize reinforcement learning (RL) to balance a ball at a specified location on a beam.

The beam was inspired by Sydney Harbour Bridge:

<p align="center">
  <img src="https://user-images.githubusercontent.com/48805713/68958201-7ba76e00-0780-11ea-8e41-464d68b56485.png">
</p>

Due to the sparse reward signal common in RL a large amount of data is needed to converge on a policy. Naturally a simulation is made to expedite the process of collecting expereience. This can be done through traditional dynamics modeling:

<p align="center">
  <img  src="https://user-images.githubusercontent.com/48805713/68958367-d5a83380-0780-11ea-9076-1ff7d7584dcb.png">
</p>

The env.py file builds such a simulation to use as a virtual enviroment.


