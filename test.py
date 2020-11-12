# Imports
from env.BeamEnv import BeamEnv
from q_learning_agent import QLearner
import numpy as np
from prettytable import PrettyTable, ALL

# Hyperparameters
save_q_table = True
filename = 'q_table_small2'
gpu = True
render = False
NUMBER_OF_EPISODES = 1e10
SAVE_EVERY_N_EPISODES = 1000

TIME_STEP = 0.25
MAX_TIME  = 15
MAX_STEPS = MAX_TIME / TIME_STEP
EPSILON   = 1 / MAX_STEPS
pretrained_q_table = np.loadtxt('q_table_small2.csv', delimiter=',')

# Display
table = PrettyTable(["Hyperparameters", "Settings"])

table.add_row(["Use GPU", gpu])
table.add_row(["Save Q-Table", save_q_table])
table.add_row(["Render Episodes", render])
table.add_row(["Number of Episodes", NUMBER_OF_EPISODES])
table.add_row(["Max Steps per Episode", MAX_STEPS])
table.add_row(["Explorations per Episode", str(EPSILON * MAX_STEPS)])
print(table)

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Initialize Environment and Agent
env   = BeamEnv(obs_low_bounds  = np.array([0,     0,  1.18e10, -30]),
                obs_high_bounds = np.array([6,     6, -1.18e10,  30]), 
                obs_bin_sizes   = np.array([1,  0.25,      20,   5]),
                TIME_STEP = TIME_STEP)

agent = QLearner(env, 
                 learning_rate   = 0.01,
                 discount_factor = 0.95,
                 pretrained_q_table = None)

# Train
num_successes = 0
percent_of_successes = []
for episode in range(int(NUMBER_OF_EPISODES)):

    # Save Q- Table
    if episode % (SAVE_EVERY_N_EPISODES) == 0:
        print("Episode: " + str(episode) + " Percent of Successes: " + str(num_successes/SAVE_EVERY_N_EPISODES))
        percent_of_successes.append(num_successes/SAVE_EVERY_N_EPISODES)
        num_successes = 0

        if save_q_table:
            np.savetxt(filename + ".csv", agent.q_table, delimiter = "," )
            np.savetxt(filename + "_performance.csv", percent_of_successes, delimiter = "," )

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

# Save Q- Table
if episode % (SAVE_EVERY_N_EPISODES) == 0:
    print("Episode: " + str(episode) + " Percent of Successes: " + str(num_successes/SAVE_EVERY_N_EPISODES))
    percent_of_successes.append(num_successes/SAVE_EVERY_N_EPISODES)
    num_successes = 0

    if save_q_table:
        np.savetxt(filename + ".csv", agent.q_table, delimiter = "," )
        np.savetxt(filename + "_performance.csv", percent_of_successes, delimiter = "," )