# Imports
import numpy as np
from env.BeamEnv import BeamEnv
from q_learning_agent import QLearner

# Hyperparameters
gpu = True
save_q_table = True
NUMBER_OF_EPISODES = 1e10
MAX_STEPS = 1000
EPSILON   = 0.2

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize Environment and Agent
env   = BeamEnv()
agent = QLearner(env)

# Train
num_successes = 0
for episode in range(int(NUMBER_OF_EPISODES)):
    if episode % 1e4 == 0:
        print("Episode: " + str(episode))

    # Reset Environment
    state = env.reset()
    
    # Experience 
    done = False
    count = 0
    while (not done) and (count > MAX_STEPS):

        # Choose next action (EPSILON-Greedy)
        action = env.action_space.sample() if np.random.uniform(0, 1) < EPSILON else agent.get_action(state)

        # Next Step
        next_state, reward, done = env.step(state, action)

        # Update Q Table
        agent.update_q_table(state, action, reward, next_state)

        # Update Current State
        state = next_state

        # Update counter
        count += 1
        if done:
            num_successes += 1

print(str(num_successes) + " successes out of " + str(NUMBER_OF_EPISODES) + " episodes." )

if save_q_table:
    np.savetxt('q_table.csv', agent.q_table, delimiter = "," )