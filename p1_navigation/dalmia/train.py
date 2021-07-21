
from unityagents import UnityEnvironment
import numpy as np

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dalmia.dqn_agent import Agent
from dalmia.model import QNetwork

# plt.ion()

env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info    = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state       = env_info.vector_observations[0]
state_size  = len(state)
# agent       = Agent(state_size=state_size, action_size=action_size, qnetwork=QNetwork, update_type='double_dqn', seed=0)
agent       = Agent(state_size=state_size, action_size=action_size, qnetwork=QNetwork, update_type='dqn', seed=0)


print('Number of agents:', len(env_info.agents))
print('Number of actions:', action_size)
print('States look like:', state)
print('States have length:', state_size)

def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay):
    """
    Deep Q-learning

    Params
    ======
        n_episodes (int): number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy policy
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) to decrease epsilon
    """

    scores = []                                               # list containing scores from each episode
    scores_window = deque(maxlen=100)                         # store only the last 100 scores
    eps = eps_start                                           # initialize epsilon (for epsilon-greedy policy)

    for i_episode in range(1, n_episodes + 1):                # run n_episodes
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        state = env_info.vector_observations[0]               # get the initial state
        score = 0                                             # initialize the score

        for t in range(max_t):                                   # run for maximum of max_t timesteps
            action     = agent.act(state, eps)                   # select the action
            env_info   = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]         # get the state
            reward     = env_info.rewards[0]                     # get the reward
            done       = env_info.local_done[0]                  # whether the episode is complete or not

            agent.step(state, action, reward, next_state, done)  # train the agent
            state  = next_state                                  # update the state
            score += reward                                      # update the score

            if done:                                             # break if episode is complete
                break

        scores_window.append(score)                # update the window of scores
        scores.append(score)                       # update the list of scores
        eps = max(eps_end, eps * eps_decay)        # modify epsilon
        average_score = np.mean(scores_window)
        # print('\rEpisode {} \tAverage score: {: .2f}'.format(i_episode, average_score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {} \tAverage score: {: .2f}'.format(i_episode, average_score))

        if average_score >= 13:      # check if environment is solved
            print('\nEnvironment solved in {: d} episodes!\tAverage Score: {: .2f}'.format(i_episode - 100, average_score))
            torch.save(agent.qnetwork_local.state_dict(), 'ddqn.pth')
            break

    return scores


if __name__ == '__main__':
    n_episodes = 5000
    max_t = 2000
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 0.995
    scores = dqn(n_episodes, max_t, eps_start, eps_end, eps_decay)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('results/dddqn_new_scores.png', bbox_inches='tight')
    plt.show()
