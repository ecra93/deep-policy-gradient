import tensorflow as tf
import numpy as np

import gym


ENVIRONMENT = "LunarLander-v2"
SEED = 1 # to get same initialization for game state and network weights

N_TRAINING_EPISODES = 1
N_EPISODES_TO_RENDER = 10
N_EPISODES_TILL_RENDER = N_TRAINING_EPISODES - N_EPISODES_TO_RENDER

DISCOUNT_RATE = 0.99
LEARNING_RATE = 0.02


class Agent:

    def __init__(self, state_shape, n_actions):

        # used to determine the shape of the network layers
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.sess = tf.Session()
        self.initialize_policy_network()
        #self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        #self.load_network()

    def act(self, s):
        policy = self.sess.run(self.y_, feed_dict={self.x:[s]})[0]
        action = np.random.choice(len(policy), p=policy)
        return action

    def initialize_policy_network(self):
        
        # input placeholders
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32,
                shape=[None]+[self.state_shape])

        # hidden layers layers
        initializer = tf.contrib.layers.xavier_initializer(seed=SEED)
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
            self.y_ = tf.nn.softmax(tf.matmul(fc2, Wo) + bo)

        # loss function
        with tf.name_scope("loss"):
            self.y = tf.placeholder(tf.float32, shape=[None,
                    self.n_actions])
            self.r = tf.placeholder(tf.float32, shape=[None, ])
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.y_, labels=self.y)
            self.loss = tf.reduce_mean(neg_log_prob * self.r)

        # optimizer
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                LEARNING_RATE).minimize(self.loss)
        

    def train(self, transitions, episode_no):

        # transpose transitions to slice each set of elements
        transitions = np.transpose(transitions)

        # grab the input states
        s = transitions[0]

        # grab the input actions, convert to one-hot
        a = transitions[1]
        a_one_hot = np.zeros([a.size, self.n_actions])
        a_one_hot[np.arange(a.size), a.astype(int)] = 1.0

        # grab the rewards
        r = transitions[2]

        # we also need to convert the rewards discounted
        discounted_r = np.zeros_like(r)
        accum = 0
        for i in reversed(range(r.size)):
            accum = accum * DISCOUNT_RATE + r[i]
            discounted_r[i] = accum
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

        # run the training step
        episode_loss, _ = self.sess.run([self.loss, self.optimizer],
            feed_dict={
                self.x: list(s),
                self.y: a_one_hot,
                self.r: discounted_r
            })

        print("===================================================")
        print("Completed Episode:", episode_no)
        print("Episode Loss:", episode_loss)
        print("Episode Reward:", np.sum(r))
        print("===================================================\n")

        # put code to save network here
        # ...

    
if __name__ == "__main__":

    # launch environment and agent
    env = gym.make(ENVIRONMENT).unwrapped
    env.seed(SEED)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape[0]
    agent = Agent(state_shape, n_actions)
    
    # run training loop
    max_reward = 0.0
    for episode in range(N_TRAINING_EPISODES+N_EPISODES_TO_RENDER):

        # restart the game
        s = env.reset()
        transitions = []
        done = False

        while not done:

            # render the environment if finished training
            if episode > N_TRAINING_EPISODES: env.render()

            # a = env.action_space.sample()
            # agent chooses action
            a = agent.act(s)

            # action is executed, causing state transition
            s_, r, done, _ = env.step(a)
            transitions.append([s, a, r, s_, done])

            # set new transition to current
            s = s_

            # compute episode reward so far
            """
            episode_reward = np.sum(np.transpose(transitions)[2])
            if episode_reward < -250:
                done = True
            """

        # train the agent at the end of the episode
        agent.train(transitions, episode)

        # print episode reward if exceeds current max
        episode_reward = np.sum(np.transpose(transitions)[2])
        if episode_reward > max_reward:
            max_reward = episode_reward
            print("###NEW MAX REWARD###", max_reward )
