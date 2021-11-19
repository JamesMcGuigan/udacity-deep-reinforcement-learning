#!/usr/bin/env python
# This is the main training loop

import os
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from src.v1_handcoded.libs.contextmanager import capture
from src.v1_handcoded.libs.env import get_env_state_action_agents_size
from src.v2_dqn.ReplayBuffer import Experience
from src.v4_ddpg_bipedal.ddpg_agent import DDPGAgent


def ddpg(n_episodes=2000, max_t=1001):
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))  # ensure cwd in project root
    # env = UnityEnvironment(file_name='./unity/Reacher_Linux_NoVis/Reacher.x86_64')
    env = UnityEnvironment(file_name='./unity/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
    # env = UnityEnvironment(file_name='./unity/Crawler_Linux_NoVis/Crawler.x86_64')
    state_size, action_size, num_agents = get_env_state_action_agents_size(env)  # == (33, 4, 20)
    agent = DDPGAgent(state_size, action_size, random_seed=0)

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        brain_name = env.brain_names[0]
        env_info   = env.reset(train_mode=True)[brain_name]
        state      = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):

            action     = agent.act(state)

            env_info   = env.step(action)[brain_name]       # send the action to the environment
            next_state = env_info.vector_observations[0]
            reward     = env_info.rewards[0]                # get the reward
            done       = env_info.local_done[0]             # see if episode has finished

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        max_score = max(max_score, score)
        print('\rEpisode {:5d}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {:5d}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, np.mean(scores_deque), max_score))
    return scores



if __name__ == '__main__':
    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
