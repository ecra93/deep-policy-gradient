import tensorflow as tf
import numpy as np

import gym


ENVIRONMENT = "LunarLander-v2"
NUM_TRAINING_EPISODES = 2


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


class Agent:
    
    def __init__(self, state_shape, n_actions):
        self.sess = tf.Session()
        self.initialize_policy_network(state_shape, n_actions)
        #self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        #self.load_network()

    def act(self, s):
        policy = self.sess.run(self.y_, feed_dict={self.x:[s]})[0]
        action = np.random.choice(len(policy), p=policy)
        return action

    def initialize_policy_network(self, state_dims, n_output_actions):
        
        # input placeholders
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None]+[state_dims])

        # hidden layers layers
        with tf.name_scope("fully-connected-01"):
            W1 = weight_variable([state_dims, 100])
            b1 = bias_variable([100])
            fc1 = tf.matmul(self.x, W1) + b1

        # output layer
        with tf.name_scope("output"):
            Wo = weight_variable([100, n_output_actions])
            bo = bias_variable([n_output_actions])
            self.y_ = tf.nn.softmax(tf.matmul(fc1, Wo) + bo)

        # loss function
        with tf.name_scope("loss"):
            self.y = tf.placeholder(tf.float32, shape=[None,
                n_output_actions])
            self.r = tf.placeholder(tf.float32, shape=[None, ])
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.y_, labels=self.y)
            loss = tf.reduce_mean(neg_log_prob * self.r)

        # optimizer
        with tf.name_scope("optimizer"):
            
        

    def train(self, transitions):
        pass


if __name__ == "__main__":

    # launch environment and agent
    env = gym.make(ENVIRONMENT)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape[0]
    agent = Agent(state_shape, n_actions)
    
    # run training loop
    for episode in range(NUM_TRAINING_EPISODES):

        # restart the game
        s = env.reset()
        transitions = []
        done = False

        while not done:

            # render the environment
            env.render()

            # a = env.action_space.sample()
            # agent chooses action
            a = agent.act(s)

            # action is executed, causing state transition
            s_, r, done, _ = env.step(a)
            transitions.append([s, a, r, s_, done])

            # set new transition to current
            s = s_

        # train the agent at the end of the episode
        agent.train(transitions)
