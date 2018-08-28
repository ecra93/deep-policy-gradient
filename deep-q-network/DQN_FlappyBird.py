"""
12/2016-1/2017
Deep Q-Network for Flappy Bird

Substantially copied from Yenchenlin's implementation of a program with the same purpose. See:
    https://github.com/yenchenlin/DeepLearningFlappyBird

Still pretty messy... this is a work in progress.
"""


import tensorflow as tf
import numpy as np
import cv2
import random
import sys
from collections import deque
from WrappedFlappyBird import WrappedFlappyBird


LEARNING_RATE = 1e-6                # learning rate for the network optimizer
EPSILON_INITIAL = 0.5               # initial probability of executing random action
EPSILON_FINAL = 0.0001              # final probability of executing random actiton
GAMMA = 0.99                        # decay rate
BATCH_SIZE = 32
OBSERVATION_PERIODS = 100000        # periods spend observing game prior to network training
EXPLORATION_PERIODS = 2000000       # periods before initial epsilon decays to final epsilon
REPLAY_MEMORY_CAPACITY = 1000       # max number of transitions to be stored at any time

JUMP = [1, 0]                       # action vector passed to game emulator to execute jump
GLIDE = [0, 1]                      # action vector passed to game emulator to execute glide


"""
The main function... which doesn't really do much here, just executes the train command.
"""
def main():
    train_neural_network()


"""
All functions used to preprocess raw game image data.
"""
def preprocess_image(image):
    image = cv2.resize(image, (80, 80))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 355, cv2.THRESH_BINARY)
    return image


"""
All functions used to define the convolutional neural network.
"""
def convolutional_neural_network():

    # input game state tensor
    game_state = tf.placeholder(tf.float32, [None, 80, 80, 4])

    # network weights, biases, and strides
    weights = {
        "conv1" : weight_variable([8, 8, 4, 32]),
        "conv2" : weight_variable([4, 4, 32, 64]),
        "conv3" : weight_variable([3, 3, 64, 64]),
        "fc1" : weight_variable([1600, 512]),
        "output" : weight_variable([512, 2])
    }

    biases = {
        "conv1" : bias_variable([32]),
        "conv2" : bias_variable([64]),
        "conv3" : bias_variable([64]),
        "fc1" : bias_variable([512]),
        "output" : bias_variable([2])
    }

    stride = {
        "conv1" : [1, 4, 4, 1],
        "conv2" : [1, 2, 2, 1],
        "conv3" : [1, 1, 1, 1]
    }

    # relationship between layers
    conv1 = conv_layer_2d(game_state, weights["conv1"], biases["conv1"], stride["conv1"])
    conv1 = tf.nn.relu(conv1)
    conv1 = max_pool_2x2(conv1)
    conv2 = conv_layer_2d(conv1, weights["conv2"], biases["conv2"], stride["conv2"])
    conv2 = tf.nn.relu(conv2)
    conv3 = conv_layer_2d(conv2, weights["conv3"], biases["conv3"], stride["conv3"])
    conv3 = tf.nn.relu(conv3)
    conv3_flat = tf.reshape(conv3, [-1, 1600])
    fc1 = fc_layer(conv3_flat, weights["fc1"], biases["fc1"])
    fc1 = tf.nn.relu(fc1)
    predicted_action_values = fc_layer(fc1, weights["output"], biases["output"])

    return game_state, predicted_action_values


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


def conv_layer_2d(input_data, weights, biases, strides):
    return tf.nn.conv2d(input_data, weights, strides=strides,
        padding="SAME") + biases


def fc_layer(input_data, weights, biases):
    return tf.matmul(input_data, weights) + biases


def max_pool_2x2(input_data):
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


"""
All functions used to train the convolutional neural network on the game.
"""
def train_neural_network():

    # start the game
    game = WrappedFlappyBird()

    # initialize tf session
    sess = tf.InteractiveSession()

    # retrieve the convnet input and output tensors
    game_state, predicted_action_values = convolutional_neural_network()


    # cost function
    actions_oh = tf.placeholder(tf.float32, [None, 2]) # one-hot array of selected actions
    predicted_action_values_1d = tf.mul(predicted_action_values, actions_oh)
    predicted_action_values_1d = tf.reduce_sum(predicted_action_values_1d)
    actual_action_values = tf.placeholder(tf.float32, [None])
    cost = tf.reduce_mean(tf.square(actual_action_values - predicted_action_values_1d))

    # optimizier
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    
    # open memory to store transitions
    replay_memory = deque()

    # generate the first game state
    frame_t_raw, reward_t, game_over = game.frame_step(GLIDE)
    frame_t = preprocess_image(frame_t_raw)
    state_t = np.stack((frame_t, frame_t, frame_t, frame_t), axis=2)


    # load network weights
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_network_weights")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Old network weights successfully loaded.")
    else:
        print("Old network weights NOT successfully loaded.")


    # train the network
    epsilon = EPSILON_INITIAL
    t = 0
    while True:
        # get predicted action values from the initial game state
        output_t = predicted_action_values.eval(feed_dict={game_state : [state_t]})
        output_t = output_t[0]
        
        # with probability epsilon, choose a random action
        action_t = None
        if random.random() <= epsilon:
            action_t = choose_random_action()
        else:
            action_t = choose_action(output_t)

        # decay epsilon towards final value
        if epsilon > EPSILON_FINAL and t > OBSERVATION_PERIODS:
            epsilon -= (EPSILON_INITIAL-EPSILON_FINAL)/EXPLORATION_PERIODS

        # execute the selected action
        frame_t1_raw, reward_t1, game_over = game.frame_step(action_t)
        frame_t1 = preprocess_image(frame_t1_raw)

        # generate the transition
        frame_t1 = np.reshape(frame_t, (80, 80, 1))
        state_t1 = np.append(frame_t1, state_t[:, :, :3], axis=2)

        # store transition in replay memory
        replay_memory.append((state_t, action_t, reward_t, state_t1, game_over))
        if len(replay_memory) > REPLAY_MEMORY_CAPACITY:
            replay_memory.popleft()

        # if the observation period is completed, then start training
        if t > OBSERVATION_PERIODS:

            # randomly sample transitions from replay memory
            batch = random.sample(replay_memory, BATCH_SIZE)
            state_t_batch = [i[0] for i in batch]
            actions_batch = [i[1] for i in batch]
            rewards_batch = [i[2] for i in batch]
            state_t1_batch = [i[3] for i in batch]
            game_over_batch = [i[4] for i in batch]
            actual_action_values_batch = []
            output_batch = predicted_action_values.eval(
                    feed_dict={game_state : state_t1_batch})
            for i in range(len(batch)):
                if game_over_batch[i]:
                    actual_action_values_batch.append(rewards_batch[i])
                else:
                    actual_action_values_batch.append(rewards_batch[i] +
                            GAMMA*np.max(output_batch[i]))
            
            # train network weights on sample
            train_step.run(feed_dict={
                    actual_action_values : actual_action_values_batch,
                    actions_oh : actions_batch,
                    game_state : state_t_batch
                })

        # update the current game state and time iteration
        state_t = state_t1
        t += 1

        # at regular intervals, print information to console
        if t% 10000 == 0:
            saver.save(sess, "saved_networks/DQN", global_step=t)
            
            state = ""
            if t < OBSERVATION_PERIODS:
                state = "Observing"
            elif t < EXPLORATION_PERIODS:
                state = "Exploring"
            else:
                state = "Training"

            print("Weights successfully saved. Iteration", t, "State:", state)


def choose_random_action():
    if random.random() >= 0.2:
        return JUMP
    else:
        return GLIDE


def choose_action(predicted_action_values):
    if np.argmax(predicted_action_values) == 0:
        return JUMP
    else:
        return GLIDE


# execute main: run the game and train the network
main()
