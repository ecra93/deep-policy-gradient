import tensorflow as tf

from network import Network
from worker import Worker


if __name__ == "__main__":

    n_actions = 9
    n_episodes = 1
    game = "MsPacman-v0"

    with tf.Session() as sess:

        # create master
        master = Network(n_actions, sess)

        # create workers
        worker = Worker(game, master)
        worker.run(n_episodes)
