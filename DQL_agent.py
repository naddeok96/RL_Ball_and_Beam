import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
# Install tensorflow-gpu: tensorflow detects GPU here
import tensorflow as tf
from env.BeamEnv import BeamEnv


class DeepQLearning:
    """"Deep Q-learning (DQN) for a big state space env. This is the brain of the agent. 
        Tensorflow is used to build the neural network.

    Args: 
        learning_rate (float): The step size at each iteration.
    """
    def __init__(self, learning_rate):
        # Build the network to predict the correct action
        tf.reset_default_graph()

        input_dimension = 4
        hidden_dimension = 36

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_dimension])

        hidden_layer = tf.contrib.layers.fully_connected(
            inputs=self.input,
            num_outputs=hidden_dimension,
            activation_fn=tf.nn.relu)

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden_layer,
            num_outputs=2,
            activation_fn=None)

        # Sample an action according to network's output
        # use tf.multinomial and sample one action from network's output
        self.action = tf.reshape(tf.multinomial(logits, 1), [])

        # Optimization according to policy gradient algorithm
        self.action_probs = tf.nn.softmax(logits)
        log_action_probabilities = tf.log(self.action_probs)

        self.n_actions = tf.placeholder(tf.int32)

        self.rewards = tf.placeholder(tf.float32)

        ind = tf.range(0, tf.shape(log_action_probabilities)[0]) * tf.shape(log_action_probabilities)[1] + self.n_actions

        probs = tf.gather(tf.reshape(log_action_probabilities, [-1]), ind)

        loss = -tf.reduce_sum(tf.multiply(probs, self.rewards))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)  # use one of tensorflow optimizers

        self.train_op = self.optimizer.minimize(loss)

        self.saver = tf.train.Saver()
        
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.ses = tf.Session(config=config)
        self.ses.run(tf.initialize_all_variables())

    def train(self, observation, n_actions, rewards):
        self.ses.run(self.train_op, feed_dict={self.input: observation,
                                               self.n_actions: n_actions,
                                               self.rewards: rewards})

    def save(self):
        self.saver.save(self.ses, "SavedModel/")

    def load(self):
        self.saver.restore(self.ses, "SavedModel/")

    def get_action(self, observation):
        return self.ses.run(self.action, feed_dict={self.input: [observation]})

    def get_prob(self, observation):
        return self.ses.run(self.action_probs, feed_dict={self.input: observation})


epochs = 500
max_steps_per_game = 2000
games_per_epoch = 10
learning_rate = 0.1
early_stop = 0

agent = DeepQLearning(learning_rate)
game = BeamEnv()


def convert_rewards(rewards):
    return [1] * len(rewards)


dr = .99


def discount(R, discount_rate=dr):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    return discounted_r


max_state = np.array([-100, -100, -100, -100])
min_state = np.array([100, 100, 100, 100])
rewards = []
total_rewards = []
for epoch in range(epochs):
    epoch_rewards = []
    epoch_actions = []
    epoch_observations = []
    epoch_average_reward = 0
    for episode in range(games_per_epoch):
        obs = game.reset()
        step = 0
        episode_rewards = []
        episode_actions = []
        episode_observations = []
        game_over = False
        while not game_over and step < max_steps_per_game:
            step += 1
            action = agent.get_action(obs)
            episode_observations.append(obs)
            max_state = np.maximum(max_state, np.array(obs))
            min_state = np.minimum(min_state, np.array(obs))

            obs, reward, game_over = game.step(obs, action)
            episode_rewards.append(reward)
            episode_actions.append(action)

        print('Episode steps: {}'.format(len(episode_observations)))
        total_rewards.append(len(episode_observations))
        epoch_rewards.extend(discount(convert_rewards(episode_rewards)))
        epoch_observations.extend(episode_observations)
        epoch_actions.extend(episode_actions)

        epoch_average_reward += sum(episode_rewards)

    epoch_average_reward /= games_per_epoch

    print("Epoch = {}, , Average reward = {}".format(epoch, epoch_average_reward))
    
    rewards.append(epoch_average_reward)
    if epoch_average_reward >= max_steps_per_game:
        cnt += 1
    else:
        cnt = 0
    if cnt > early_stop:
        break

    normalized_rewards = (epoch_rewards - np.mean(total_rewards))
    agent.train(epoch_observations, epoch_actions, normalized_rewards)