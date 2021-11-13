import os
import re
import time
from collections import deque
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.v1_handcoded.libs.Trajectory import Experience, Trajectory


class Agent(object):
    """ Base interface class for Unity agents """

    def __init__(self, state_size, action_size, num_agents, params={}):
        self.action_size = action_size
        self.state_size  = state_size
        self.num_agents  = num_agents
        self.params      = params
        # self.actor       = self.get_actor_network()
        # self.critic      = self.get_critic_network()
        self.obs_attr    = 'vector_observations'
        self._next_idx   = 0


    def next_idx(self):
        self._next_idx += 1
        return self._next_idx


    def act(self, state, eps=0) -> int:
        pass


    def step_experience(self, experience: Experience):
        """ Agents may choose to define either step_experience() or step_trajectory() """
        (state, action, reward, next_state, done, idx) = experience
        pass


    def step_trajectory(self, trajectory: Trajectory, eps=1.0):
        """ Agents may choose to define either step_experience() or step_trajectory() """
        for experience in trajectory:
            self.step_experience(experience)


    # def get_actor_network(self):
    #     return None
    #
    #
    # def get_critic_network(self):
    #     return None


    @property
    def persisted_fields(self) -> List[str]:
        """ These are the neural network fields to be persisted upon load/save """
        return ['actor', 'critic']


    def filename(self, name='', field='', ext='pth'):
        filename = f'./models/{self.__class__.__name__}.{name}.{field}.{ext}'
        filename = re.sub(r'(\w+)\.+\1', r'\1', filename)   # remove duplicate words
        filename = re.sub(r'\.+', r'', filename)             # remove duplicate dots
        return filename


    def load(self, name=''):
        for field in self.persisted_fields:
            filename = self.filename(field, name)
            try:
                if os.path.isfile(filename):
                    getattr(self, field).load_state_dict(torch.load(filename))
                    print(  f'{self.__class__.__name__}.load(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
                else: print(f'{self.__class__.__name__}.load(): {filename} path not found')
            except Exception as e:print(f'{self.__class__.__name__}.load(): {filename} exception: {e}')


    def save(self, name=''):
        for field in self.persisted_fields:
            filename = self.filename(field, name)
            try:
                torch.save(getattr(self, field).state_dict(), filename)
                print(f'\n{self.__class__.__name__}.save(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
            except Exception as e: print(f'{self.__class__.__name__}.save(): {filename} exception: {e}')


    def train(self,
              env,
              n_episodes=2000,
              max_t=100,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995,
              score_window_size=100,
              win_score=30,
              use_future_reward=False,
    ) -> List[float]:
        time_start = time.perf_counter()
        agent  = self
        scores = []                        # list containing rewards from each episode
        try:
            print('train(', {
                "n_episodes":        n_episodes,
                "max_t":             max_t,
                "eps_start":         eps_start,
                "eps_end":           eps_end,
                "eps_decay":         eps_decay,
                "score_window_size": score_window_size,
                "win_score":         win_score,
                "obs_attr":          self.obs_attr,
                "use_future_reward": use_future_reward,
            })

            scores_window = deque(maxlen=score_window_size)  # last 100 rewards
            eps = eps_start                    # initialize epsilon
            for i_episode in range(1, n_episodes+1):
                trajectory = self.generate_trajectory(env, max_t, eps)
                score      = trajectory.get_total_reward()
                scores_window.append(score)       # save most recent score
                scores.append(score)              # save most recent score
                eps = max(eps_end, eps_decay*eps) # decrease epsilon

                agent.step_trajectory(trajectory, eps)

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
                time_taken = time.perf_counter() - time_start
                print('\nEnvironment unsolved after {:d} episodes!\tAverage Score: {:.2f} | Time: {:.0f}s'
                      .format(i_episode, np.mean(scores_window), time_taken))

        except KeyboardInterrupt as exception:
            # agent.save(filename)
            pass

        return scores


    def generate_trajectory(self, env, max_t, eps):
        agent = self
        trajectory = Trajectory()
        brain_name = env.brain_names[0]
        env_info   = env.reset(train_mode=True)[brain_name]
        state      = getattr(env_info, self.obs_attr)[0]
        for t in range(max_t):
            action     = agent.act(state, eps)
            env_info   = env.step(action)[brain_name]       # send the action to the environment
            next_state = getattr(env_info, self.obs_attr)[0]
            reward     = env_info.rewards[0]                # get the reward
            done       = env_info.local_done[0]             # see if episode has finished

            experience = Experience(state, action, reward, next_state, done, self.next_idx())
            trajectory.append( experience )

            state = next_state                         # roll over the state to next time step
            if done: break

        return trajectory


    def plot_rewards(self, rewards: List[float], model_name: str=''):
        agent      = self
        model_name = model_name or agent.__class__.__name__
        filename   = agent.filename(name=model_name, ext='.png')

        # plot the rewards
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        plt.plot(np.arange(len(rewards)), rewards)
        plt.title(f'Plot of Rewards: {model_name}')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
