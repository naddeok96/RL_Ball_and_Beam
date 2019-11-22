# Imports
from env import Beam
from agent import Balancer
from silence import shh


env = Beam()
agent = Balancer(env)
history = agent.episode()
print(history)