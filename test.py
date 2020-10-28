# Imports
import numpy as np
from env.BeamEnv import BeamEnv
from q_learning_agent import QLearner

# Hyperparameters
save_q_table = False
NUMBER_OF_EPISODES = 1e2
MAX_STEPS = 1000
EPSILON   = 0.2

# Initialize Environment and Agent
env   = BeamEnv()
agent = QLearner(env)

# Train
num_successes = 0
for episode in range(int(NUMBER_OF_EPISODES)):
    if episode % (NUMBER_OF_EPISODES / 10) == 0:
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