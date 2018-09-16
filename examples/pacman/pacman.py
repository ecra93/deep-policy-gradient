import tensorflow as tf

from network import Network, Optimizer
from worker import Worker


if __name__ == "__main__":

    n_actions = 9
    n_episodes = 1
    n_workers = 4
    game = "MsPacman-v0"

    with tf.Session() as sess:

        # create master
        network = Network(n_actions, sess)

        # create workers
        workers = []
        for _ in range(n_workers):
            worker = Worker(game, network, n_episodes)
            workers.append(worker)

        # start each of the workers
        for i in range(1, n_workers):
            workers[i].start()

        # start an optimizer
        optimizer = Optimizer(network)
        optimizer.start()

        # run worker[0] on main thread
        workers[0].is_main = True
        workers[0].run()

        # wait for the remaining workers to finish
        for i in range(1, n_workers):
            workers[i].join()

        # wait for the optimizer to wrap up
        optimizer.join()
