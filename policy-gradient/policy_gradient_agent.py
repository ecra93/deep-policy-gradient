import tensorflow as tf
import numpy as np


ENVIRONMENT = "LunarLander-v2"

N_TRAINING_EPISODES = 500
N_EPISODES_TO_RENDER = 20
N_EPISODES_TILL_RENDER = N_TRAINING_EPISODES - N_EPISODES_TO_RENDER

EPISODE_LENGTH_TILL_RENDER = 30
MAX_EPISODE_LENGTH = 120 # in seconds

DISCOUNT_RATE = 0.99
LEARNING_RATE = 0.02


class Agent:

    def __init__(self, state_shape, n_actions, discount_rate=0.99,
        learning_rate=0.01, save_path=None, load_path=None):

        # used to determine the shape of the network layers
        self.state_shape = state_shape
        self.n_actions = n_actions

        # save and load paths
        self.save_path = save_path
        self.load_path = load_path

        # episode stuff
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # training parameters
        self.discount_rate = 0.99
        self.learning_rate = 0.02

        # init the network, launch tensorflow session
        self.initialize_policy_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #  load previously saved networks if any
        self.saver = tf.train.Saver()
        if load_path:
            checkpoint = tf.train.latest_checkpoint(load_path)
            if checkpoint:
                print("Loading existing network at:", self.load_path)
                self.saver.restore(self.sess, checkpoint)

    def initialize_policy_network(self):
        
        # input placeholders
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32,
                shape=[None, self.state_shape])

        # hidden layers layers
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope("fully-connected-01"):
            W1 = tf.get_variable(name="W1", shape=[self.state_shape, 10],
                    initializer=initializer)
            b1 = tf.get_variable(name="b1", shape=[10],
                    initializer=initializer)
            fc1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        with tf.name_scope("fully-connected-02"):
            W2 = tf.get_variable(name="W2", shape=[10, 10],
                    initializer=initializer)
            b2 = tf.get_variable(name="b2", shape=[10],
                    initializer=initializer)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

        # output layer
        with tf.name_scope("output"):
            Wo = tf.get_variable(name="Wo", shape=[10, self.n_actions],
                    initializer=initializer)
            bo = tf.get_variable(name="bo", shape=[self.n_actions],
                    initializer=initializer)
            y_pre_softmax = tf.matmul(fc2, Wo) + bo
            self.y_ = tf.nn.softmax(y_pre_softmax)

        # loss function
        with tf.name_scope("loss"):
            self.y = tf.placeholder(tf.float32, shape=[None,
                    self.n_actions])
            self.r = tf.placeholder(tf.float32, shape=[None, ])
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
                    logits=y_pre_softmax, labels=self.y)
            self.loss = tf.reduce_mean(neg_log_prob * self.r)

        # optimizer
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                LEARNING_RATE).minimize(self.loss)
        
    def act(self, s):
        policy = self.sess.run(self.y_, feed_dict={self.x:[s]})[0]
        action = np.random.choice(len(policy), p=policy)
        return action

    def save_transition(self, s, a, r):
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)

    def train(self, episode_no):

        # convert actions to one-hot
        a_one_hot = np.zeros([len(self.episode_actions), self.n_actions])
        a_one_hot[np.arange(len(self.episode_actions)),
            list(map(int, self.episode_actions))] = 1.0

        # discount rewards
        r_discounted = np.zeros_like(self.episode_rewards)
        accum = 0.0
        for i in reversed(range(len(self.episode_rewards))):
            accum = accum * self.discount_rate + self.episode_rewards[i]
            r_discounted[i] = accum
        r_discounted -= np.mean(r_discounted)
        r_discounted /= np.std(r_discounted)

        # run the training step
        episode_loss, _ = self.sess.run([self.loss, self.optimizer],
            feed_dict={
                self.x: self.episode_states,
                self.y: a_one_hot,
                self.r: r_discounted
            })

        print("===================================================")
        print("Completed Episode:", episode_no)
        print("Episode Loss:", episode_loss)
        print("Episode Reward:", np.sum(self.episode_rewards))
        print("===================================================")

        # reset the episode state, action, reward accumulators
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # put code to save network here
        if self.save_path:
            self.saver.save(self.sess, self.save_path)
        
