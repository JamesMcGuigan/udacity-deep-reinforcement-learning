#!/usr/bin/env python
#
# This file represents the main training loop
import os
from collections import deque
import time

import numpy as np
from matplotlib import pyplot as plt

from unityagents import UnityEnvironment
from libs.env import get_env_state_action_agents_size
from src.agents.RandomAgent import RandomAgent
from src.libs.Trajectory import Trajectory, Experience
from src.libs.contextmanager import capture


def train(env, agent,
          n_episodes=200,
          max_t=100,
          eps_start=1.0,
          eps_end=0.01,
          eps_decay=0.995,
          score_window_size=100,
          win_score=30,
          obs_attr='vector_observations',
          use_future_reward=False,
    ):
    time_start = time.perf_counter()
    scores = []                        # list containing scores from each episode
    try:
        print('train(', {
            "n_episodes":        n_episodes,
            "max_t":             max_t,
            "eps_start":         eps_start,
            "eps_end":           eps_end,
            "eps_decay":         eps_decay,
            "score_window_size": score_window_size,
            "win_score":         win_score,
            "obs_attr":          obs_attr,
            "use_future_reward": use_future_reward,
        })
        brain_name  = env.brain_names[0]
        # brain       = env.brains[brain_name]
        # action_size = brain.vector_action_space_size        # action_size == 4

        scores_window = deque(maxlen=score_window_size)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]

            trajectory = Trajectory()
            score = 0
            state = getattr(env_info, obs_attr)[0]
            for t in range(max_t):
                action     = agent.act(state, eps)
                env_info   = env.step(action)[brain_name]       # send the action to the environment
                next_state = getattr(env_info, obs_attr)[0]
                reward     = env_info.rewards[0]                # get the reward
                done       = env_info.local_done[0]             # see if episode has finished

                experience = Experience(state, action, reward, next_state, done)
                if use_future_reward: trajectory.append( experience )
                else:                 agent.step( experience )

                state      = next_state                         # roll over the state to next time step
                score     += reward

                if done: break

            if use_future_reward:
                for experience in trajectory.with_future_rewards(eps):
                    agent.step( experience )

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            # print('\rEpisode {:4d}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {:4d}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            time_taken = time.perf_counter() - time_start
            if np.mean(scores_window) >= win_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} | Time: {:.0f}s'
                      .format(i_episode-100, np.mean(scores_window), time_taken))
                agent.save()
                break
        else:
            print('\nEnvironment unsolved after {:d} episodes!\tAverage Score: {:.2f} | Time: {:.0f}s'
                  .format(i_episode, np.mean(scores_window), time_taken))

    except KeyboardInterrupt as exception:
        # agent.save(filename)
        pass

    return scores


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # ensure cwd in project root
    env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')
    # env = UnityEnvironment(file_name='./Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
    # env = UnityEnvironment(file_name='./Crawler_Linux_NoVis/Crawler.x86_64')
    state_size, action_size, num_agents = get_env_state_action_agents_size(env)  # == (33, 4, 20)

    configs = [
        { "model_name": "random", "agent_class": RandomAgent, "params": {} },
    ]
    for config in configs:
        with capture() as stdout:
            print(f'\nconfig: {config}')
            scores = []
            model_name  = config['model_name']
            agent_class = config['agent_class']
            agent       = agent_class(state_size, action_size, num_agents,
                                      params=config.get('params',{}), **config.get('kwargs',{}))
            # agent.load()
            scores += train(env, agent)
            # agent.save()


        filename = agent.filename().replace('.pth', '')

        print("\n".join(stdout))
        with open(f'{filename}.log', 'w') as f:
            f.write("\n".join(stdout))

        # plot the scores
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.title(f'Plot of Rewards: {model_name}')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(f'{filename}.png', bbox_inches='tight')
        plt.show()
