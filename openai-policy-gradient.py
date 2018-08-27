import gym

NUM_TRAINING_EPISODES = 1


class Agent:
    
    def __init__(self, state_shape, n_actions):
        pass

    def act(self, s):
        return 1

    def train(self, transitions):
        pass


if __name__ == "__main__":

    # launch environment and agent
    env = gym.make("LunarLander-v2")
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    agent = Agent(state_shape, n_actions)
    
    # run training loop
    for episode in range(NUM_TRAINING_EPISODES):

        # restart the game
        s = env.reset()
        transitions = []
        done = False

        while not done:

            # render the environment
            env.render()

            # a = env.action_space.sample()
            # agent chooses action
            a = agent.act(s)

            # action is executed, causing state transition
            s_, r, done, _ = env.step(a)
            transitions.append([s, a, r, s_, done])

            # set new transition to current
            s = s_

        # train the agent at the end of the episode
        agent.train(transitions)
