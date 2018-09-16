import tensorflow as tf
import numpy as np
import threading

class Network:
    """
    Global network.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.initialize_network()
        self.transitions = []
        self.lock = threading.Lock()

    def initialize_network(self):
        # define the network architecture here

        # (1) define input state placeholder, input_shape=(84, 84, 1)

        with tf.name_scope("input"):
            X = tf.placeholder(tf.float32, shape=[None, 84, 84, 1])

        with tf.name_scope("conv_layer-1") as scope:
            k11 = tf.get_variable("k11", shape=[8,8,1,16])
            b1 = tf.get_variable("b1", shape=[16])
            c1 = tf.nn.conv2d(X, k11, strides=[1,1,1,1], padding="SAME")
            conv_with_bias1 = tf.add(c1, b1)
            conv1 = tf.nn.relu(conv_with_bias1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")   

        with tf.name_scope("conv-layer-2") as scope:
            k2 = tf.get_variable("k2", shape=[4,4,16,32])
            b2 = tf.get_variable("b2", shape=[32])
            c2 = tf.nn.conv2d(pool1, k2, strides=[1,1,1,1], padding="SAME")
            conv_with_bias2 = tf.add(c2, b2)
            conv2 = tf.nn.relu(conv_with_bias2)
            pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

            # (2) define network architecture, 2 fully connected layers, L1 has 100 cells, L2 has 50 cells, output has n_actions

        with tf.name_scope("fc-layer-1"):
            x_flat = tf.reshape(pool2, shape=[-1,21*21*32])
            w3 = tf.get_variable(name="w3", shape=[21*21*32, 100])
            b3 = tf.get_variable(name="b3", shape=[100])
            fc1 = tf.nn.relu(tf.matmul(x_flat, w3) + b3)

        with tf.name_scope("output"):
            w4 = tf.get_variable(name="w4", shape=[100, self.n_actions])
            b4 = tf.get_variable(name="b4", shape=[self.n_actions])
            y_ = tf.matmul(fc1, w4) + b4
            policy = tf.nn.softmax(y_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            p = sess.run(policy, feed_dict={X:[np.ones(shape=(84,84,1), dtype=float)]})
            print(p) 

    def train_network(self):
        # train network here
        pass

    def choose_action(self, state):
        # stub: currently just chooses a random action
        return np.random.randint(self.n_actions)

    def store_transitions(transitions):
        self.lock.acquire()
        self.transitions.append(transitions)
        self.lock.release()
