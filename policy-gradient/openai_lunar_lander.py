import numpy as np
import time
import gym

from policy_gradient_agent import Agent


ENVIRONMENT = "LunarLander-v2"
RENDER_ENVIRONMENT = False
REWARD_TO_TRIGGER_RENDER = 165
SEED = 1

N_TRAINING_EPISODES = 500
N_OBSERVATION_EPISODES = 20
MAX_EPISODE_DURATION = 120

SAVE_PATH = "./lunar_lander/"


# initialize environment
env = gym.make(ENVIRONMENT).unwrapped
env.seed(SEED)
n_actions = env.action_space.n
state_shape = env.observation_space.shape[0]

# initialize agent
agent = Agent(state_shape, n_actions, discount_rate=0.99, 
    learning_rate=0.02, save_path=SAVE_PATH, load_path=SAVE_PATH)

# run the training loop
max_reward = -250.0
for episode_no in range(N_TRAINING_EPISODES + N_OBSERVATION_EPISODES):
    
    # reset the game
    s = env.reset()
    done = False

    # run the episode
    start_t = time.clock()
    while not done:

        # render if switched on
        if episode_no > N_TRAINING_EPISODES: env.render()

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

        # if too much time has elapsed, then move on
        current_t = time.clock()
        if current_t - start_t > MAX_EPISODE_DURATION:
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
