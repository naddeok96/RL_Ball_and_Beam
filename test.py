# Imports
from env.BeamEnv import BeamEnv
from q_learning_agent import QLearner
import numpy as np

# Hyperparameters
save_q_table = True
gpu = True
render = False
NUMBER_OF_EPISODES = 1e10
MAX_STEPS = 150
EPSILON   = 0.1
# pretrained_q_table = np.loadtxt('pretrained_q_tables/q_table_11_6.csv', delimiter=',')

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Initialize Environment and Agent
env   = BeamEnv(obs_low_bounds  = np.array([0,   0,  1.18e10, -30]),
                obs_high_bounds = np.array([6,   6, -1.18e10,  30]), 
                obs_bin_sizes   = np.array([1, 0.5,        6,   5]))

agent = QLearner(env, 
                 learning_rate = 0.20,
                 discount_factor = 0.80,
                 pretrained_q_table = None)

# Train
num_successes = 0
for episode in range(int(NUMBER_OF_EPISODES)):
    if episode % (500) == 0:
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
        next_state, reward, done = env.step(state, action, render = render)

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
    np.savetxt('q_table_small.csv', agent.q_table, delimiter = "," )