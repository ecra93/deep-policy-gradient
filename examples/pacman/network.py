import tensorflow as tf
import numpy as np
import threading

class Network:
    """
    Global network.
    """
    def __init__(self, n_actions, sess):
        self.n_actions = n_actions
        self.episodes = []
        self.lock = threading.Lock()

        self.sess = sess
        self.initialize_network()
        self.sess.run(tf.global_variables_initializer())


    def initialize_network(self):

        with tf.name_scope("state-input"):
            self.X = tf.placeholder(tf.float32, shape=[None,84,84,4])

        with tf.name_scope("conv_layer-1") as scope:
            k1 = tf.get_variable("k1", shape=[8,8,4,16])
            b1 = tf.get_variable("b1", shape=[16])
            c1 = tf.nn.conv2d(self.X, k1, strides=[1,1,1,1], padding="SAME")
            conv_with_bias1 = tf.add(c1, b1)
            conv1 = tf.nn.relu(conv_with_bias1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],
                        strides=[1,2,2,1], padding="SAME")   

        with tf.name_scope("conv-layer-2") as scope:
            k2 = tf.get_variable("k2", shape=[4,4,16,32])
            b2 = tf.get_variable("b2", shape=[32])
            c2 = tf.nn.conv2d(pool1, k2, strides=[1,1,1,1], padding="SAME")
            conv_with_bias2 = tf.add(c2, b2)
            conv2 = tf.nn.relu(conv_with_bias2)
            pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],
                        strides=[1,2,2,1], padding="SAME")

        with tf.name_scope("fc-layer-1"):
            x_flat = tf.reshape(pool2, shape=[-1,21*21*32])
            w3 = tf.get_variable(name="w3", shape=[21*21*32, 100])
            b3 = tf.get_variable(name="b3", shape=[100])
            fc1 = tf.nn.relu(tf.matmul(x_flat, w3) + b3)

        with tf.name_scope("policy-output"):
            w4 = tf.get_variable(name="w4", shape=[100, self.n_actions])
            b4 = tf.get_variable(name="b4", shape=[self.n_actions])
            y_ = tf.matmul(fc1, w4) + b4
            self.policy = tf.nn.softmax(y_)


    def train_network(self):
        # train network here
        pass


    def choose_action(self, state):
        policy = self.sess.run(self.policy, feed_dict={self.X:state})[0]
        action = np.random.choice(len(policy), p=policy)
        return action


    def store_transitions(self, s0, a, s1, r_discounted):
        self.lock.acquire()
        self.episodes.append((s0, a, s1, r_discounted))
        self.lock.release()
