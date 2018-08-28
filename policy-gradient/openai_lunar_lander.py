from policy_gradient_agent import Agent
import numpy as np
import gym

ENVIRONMENT = "LunarLander-v2"
SEED = 1

N_TRAINING_EPISODES = 500

# initialize environment
env = gym.make(ENVIRONMENT).unwrapped
env.seed(SEED)
n_actions = env.action_space.n
state_shape = env.observation_space.shape[0]

# initialize agent
agent = Agent(state_shape, n_actions, discount_rate=0.99, 
    learning_rate=0.02)

# run the training loop
max_reward = -250.0
for episode_no in range(N_TRAINING_EPISODES):
    
    # reset the game
    s = env.reset()
    done = False

    # run the episode
    while not done:

        # agent chooses an action
        a = agent.act(s)

        # execute action, cause state transition
        s_, r, done, _ = env.step(a)

        # save transition
        agent.save_transition(s, a, r)

        # compute reward accumulated so far
        episode_reward = np.sum(np.transpose(agent.episode_rewards))
        if episode_reward < -250.0:
            done = True

        # transition to next state
        s = s_

    # train the agent
    agent.train(episode_no)

    # print out the max reward accumulated so far
    if episode_reward > max_reward:
        max_reward = episode_reward
    print("Max Reward:", max_reward)
    print("===================================================\n")
