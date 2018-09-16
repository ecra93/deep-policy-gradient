from network import Network
from worker import Worker

def main():

    num_actions = 10
    # create master
    master = Network(num_actions)
    
    # create workers
    worker1 = Worker("will", "MsPacman-v0", master)
    worker1.start()



if __name__ == "__main__":
    main()
