
'''
This will build an animation of an episode
'''
from env import Beam
from agent import Balancer

env = Beam()
agent = Balancer(env)

agent.episode()