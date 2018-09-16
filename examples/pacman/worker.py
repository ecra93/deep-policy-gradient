from threading import Thread
import time
import gym

class Worker(Thread):
    def __init__(self, name, env_string, network):
        Thread.__init__(self)
        self.name = name
        self.env = gym.make(env_string)
        self.network = network
        self.transitions = []

    def play_episode(self):

        # reset environment for each iteration
        s0 = self.env.reset()

        # run the game for an episode
        done = False
        while not done:
            # choose an action
            a =  self.network.choose_action(s0)

            # execute action
            s1, r1, done, info = self.env.step(a)

            # log transition
            transition = (s0, a, s1, r1)
            self.transitions.append(transition)

            # display action on screen
            self.env.render()

            # update initial state
            s0 = s1

    def run(self, num_episodes):
        # start a new thread for each episode
        for _ in range(num_episodes):
            self.play_episode()
        
        # shutdown worker
        self.env.close()

    def send_to_master(self):
        # send transitions to master
        self.network.store_stransitions(transitions)
