import gym

class Agent:
    
    def __init__(self):
        pass

    def act(self, s):
        pass

    def train(self, transitions):
        pass


if __name__ == "__main__":

    # launch environment and agent
    env = gym.make("LunarLander-v2")
    agent = Agent()
    
    # run training loop
    s = env.reset()
    for episode in range(NUM_TRAINING_EPISODES)::

        # a = env.action_space.sample()
        # agent chooses action
        a = agent.act(s)

        # action is executed, causing state transition
        s_, r, done, _ = env.step(a)
        transitions.append([s, a, r, s_, done])

        # if the game hasn't finished, keep going
        if not done:
            s = s_

        # if the game has finished, stop and train
        else:
            env.reset()
            agent.train(transitions)
            transitions = []
