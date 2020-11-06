# Imports
from env.BeamEnv import BeamEnv
from q_learning_agent import QLearner
import numpy as np

# Hyperparameters
save_q_table = False
gpu = False
NUMBER_OF_EPISODES = 2
MAX_STEPS = 200
EPSILON   = 0
pretrained_q_table = np.loadtxt('pretrained_q_tables/q_table_11_6.csv', delimiter=',')

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Initialize Environment and Agent
env   = BeamEnv()
agent = QLearner(env, 
                 pretrained_q_table = pretrained_q_table)

# Train
num_successes = 0
for episode in range(int(NUMBER_OF_EPISODES)):
    if episode % (1e3) == 0:
        print("Episode: " + str(episode))

        if save_q_table:
            np.savetxt('q_table.csv', agent.q_table, delimiter = "," )

    # Reset Environment
    state = env.reset()

    # Experience 
    done = False
    count = 0
    while (not done) and (count < MAX_STEPS):
        # Choose next action (EPSILON-Greedy)
        action = env.action_space.sample() if np.random.uniform(0, 1) < EPSILON else agent.get_action(state)

        # Next Step
        next_state, reward, done = env.step(state, action, render = True)

        # Count Successes
        if done:
            num_successes += 1

        # Update Q Table
        agent.update_q_table(state, action, reward, next_state)

        # Update Current State
        state = next_state

        # Update counter
        count += 1

print(str(num_successes) + " successes out of " + str(NUMBER_OF_EPISODES) + " episodes." )

if save_q_table:
    np.savetxt('q_table.csv', agent.q_table, delimiter = "," )