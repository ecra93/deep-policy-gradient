from network import Network
from worker import Worker

def main():

    num_actions = 9
    num_episodes = 3
    # create master
    master = Network(num_actions)
    
    # create workers
    worker1 = Worker("w1", "MsPacman-v0", master)
    worker1.run(num_episodes)

if __name__ == "__main__":
    main()
