from threading import Thread
import time
import gym

import processing

class Worker(Thread):
    def __init__(self, game, network):
        Thread.__init__(self)
        self.env = gym.make(game)
        self.network = network

    def play_episode(self):

        # reset environment for each iteration
        x0 = self.env.reset()
        x0 = process.process_raw_state(x0)
        s0 = (x0, x0, x0, x0)

        # store episode transitions
        episode_s0 = []
        episode_a = []
        episode_s1 = []
        episode_r = []

        # run the game for an episode
        done = False
        while not done:
            # choose an action
            a = self.network.choose_action(s0)

            # execute action
            x1, r1, done, info = self.env.step(a)
            x1 = process.process_raw_state(x1)
            s1 = (x1, s0[0], s[1], s[2])

            # store transition
            episode_s0.append(s0)
            episode_a.append(a)
            episode_s1.append(s1)
            episode_r.append(r)

            # display action on screen
            self.env.render()

            # update initial state
            s0 = s1

        # send transitions over to the network
        self.send_transitions_to_network(episode_s0, episode_a,
                episode_s1, episode_r)

    def run(self, n_episodes):
        # play episode n times
        for _ in range(n_episodes):
            self.play_episode()
        
        # gracefully shutdown environment
        self.env.close()

    def send_tranitions_to_network(self, s0, a, s1, r):
        r_discounted = process.process_discounted_rewards(r)
        self.network.store_stransitions(s0, a, s1, r_discounted)
