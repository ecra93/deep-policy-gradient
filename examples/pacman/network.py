import threading
import numpy

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
        pass

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
        
