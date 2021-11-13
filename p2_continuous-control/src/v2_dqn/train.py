import os
import time
from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from src.v2_dqn.ReplayBuffer import Experience
from src.v2_dqn.dqn_agent import DQNPolicyAgent
from src.v2_dqn.model import QPolicyNetwork
from src.v1_handcoded.libs.contextmanager import capture


def train_dqn(
        env, agent, filename,
        obs_attr='vector_observations',  # vector_observations || visual_observations
        # obs_attr='visual_observations',  # vector_observations || visual_observations
        n_episodes=2000,
        max_t=100,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        score_window_size=100, win_score=13,
        exit_after_first_reward=False,
        use_future_reward=False,
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    time_start = time.perf_counter()
    scores = []                        # list containing scores from each episode
    try:
        print('train_dqn(', {
            # 'env': env,
            # 'agent': agent,
            # 'filename': filename,
            'obs_attr': obs_attr,
            'n_episodes': n_episodes,
            'max_t': max_t,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'score_window_size': score_window_size,
            'win_score': win_score,
            'exit_after_first_reward': exit_after_first_reward,
            'use_future_reward': use_future_reward,
        }, ')')

        brain_name  = env.brain_names[0]
        # brain       = env.brains[brain_name]
        # action_size = brain.vector_action_space_size        # action_size == 4

        scores_window = deque(maxlen=score_window_size)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            if obs_attr == 'vector_observations': state = env_info.vector_observations[0]
            if obs_attr == 'visual_observations': state = env_info.visual_observations[0]  # is always empty list

            trajectory = []
            score = 0
            state = getattr(env_info, obs_attr)[0]
            for t in range(max_t):
                idx        = i_episode * max_t + t
                action     = agent.act(state, eps)
                env_info   = env.step(action)[brain_name]       # send the action to the environment
                next_state = getattr(env_info, obs_attr)[0]
                reward     = env_info.rewards[0]                # get the reward
                done       = env_info.local_done[0]             # see if episode has finished

                if use_future_reward:
                    trajectory.append( Experience(state, action, reward, next_state, done, idx) )
                else:
                    agent.step(state, action, reward, next_state, done, idx)

                state      = next_state                         # roll over the state to next time step
                score     += reward

                if exit_after_first_reward and score != 0: break  # keep going until at least one yellow banana
                if done: break

            # As rewards are sparse, using decayed future reward allows training on events leading upto a banana
            # This tends to speed up very early stage training, but tends to converge to lower average scores
            # Using eps as the decay factor seems to work as well as not using future_rewards
            #   use_future_reward = 1.0 | Episode 2000	Average Score: 4.33
            #   use_future_reward = 0.9 | Episode 2000	Average Score: 3.56
            #   use_future_reward = 0.0 | Episode 2000	Average Score: 5.51
            #   use_future_reward = eps | Episode 2000	Average Score: 5.36
            if use_future_reward:
                future_reward = 0
                for n in range(len(trajectory)):
                    state, action, reward, next_state, done = trajectory[-n]
                    future_reward *= eps   # was: *= use_future_reward
                    future_reward += reward
                    trajectory[-n] = Experience(state, action, future_reward, next_state, done)

                for experience in trajectory:
                    state, action, reward, next_state, done = experience
                    agent.step(state, action, reward, next_state, done)


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
                agent.save(filename)
                break
        else:
            print('\nEnvironment unsolved after {:d} episodes!\tAverage Score: {:.2f} | Time: {:.0f}s'
                  .format(i_episode, np.mean(scores_window), time_taken))

    except KeyboardInterrupt as exception:
        # agent.save(filename)
        pass

    return scores



if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))  # ensure cwd in project root
    env   = UnityEnvironment(file_name="./unity/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64")
    state_size, action_size = DQNPolicyAgent.get_env_state_action_size(env)  #  state_size == 37, action_size == 4

    configs = [
        { "model_name": "dqn",           "model_class": QPolicyNetwork },
        # { "model_name": "dueling_dqn",   "model_class": DuelingQNetwork },

        # { "model_name": "dqn@64x64",     "model_class": QNetwork,  "kwargs": { "fc1_units": 64, "fc2_units": 64 } },
        # { "model_name": "dqn@64x32",     "model_class": QNetwork,  "kwargs": { "fc1_units": 64, "fc2_units": 32 } },
        # { "model_name": "dqn@32x32",     "model_class": QNetwork,  "kwargs": { "fc1_units": 32, "fc2_units": 32 } },
        # { "model_name": "dqn@32x16",     "model_class": QNetwork,  "kwargs": { "fc1_units": 32, "fc2_units": 16 } },
        # { "model_name": "dqn@16x16",     "model_class": QNetwork,  "kwargs": { "fc1_units": 16, "fc2_units": 16 } },
        # { "model_name": "dqn@16x8",      "model_class": QNetwork,  "kwargs": { "fc1_units": 16, "fc2_units":  8 } },
        # { "model_name": "dqn@8x8",       "model_class": QNetwork,  "kwargs": { "fc1_units":  8, "fc2_units":  8 } },

        # { "model_name": "dqn@lr=1e-2",   "model_class": QNetwork, "kwargs": { "LR": 1e-2 } },
        # { "model_name": "dqn@lr=1e-3",   "model_class": QNetwork, "kwargs": { "LR": 1e-3 } },
        # { "model_name": "dqn@lr=1e-4",   "model_class": QNetwork, "kwargs": { "LR": 1e-4 } },
        # { "model_name": "dqn@lr=1e-5",   "model_class": QNetwork, "kwargs": { "LR": 1e-5 } },

        # { "model_name": "dqn@tau=1e-1",    "model_class": QNetwork, "kwargs": { "TAU": 1e-1 } },
        # { "model_name": "dqn@tau=1e-2",    "model_class": QNetwork, "kwargs": { "TAU": 1e-2 } },
        # { "model_name": "dqn@tau=1e-3",    "model_class": QNetwork, "kwargs": { "TAU": 1e-3 } },
        # { "model_name": "dqn@tau=1e-4",    "model_class": QNetwork, "kwargs": { "TAU": 1e-4 } },
        # { "model_name": "dqn@tau=1e-5",    "model_class": QNetwork, "kwargs": { "TAU": 1e-5 } },
        #
        # { "model_name": "dqn@gamma=1",     "model_class": QNetwork, "kwargs": { "GAMMA": 1     } },
        # { "model_name": "dqn@gamma=0.99",  "model_class": QNetwork, "kwargs": { "GAMMA": 0.99  } },
        # { "model_name": "dqn@gamma=0.9",   "model_class": QNetwork, "kwargs": { "GAMMA": 0.9   } },
        # { "model_name": "dqn@gamma=0.5",   "model_class": QNetwork, "kwargs": { "GAMMA": 0.5   } },
        # { "model_name": "dqn@gamma=0",     "model_class": QNetwork, "kwargs": { "GAMMA": 0     } },

        # { "model_name": "dqn@memory_type=subtree", "model_class": QNetwork, "kwargs": { "memory_type": 'subtree' } },
        # { "model_name": "dqn@memory_type=random",  "model_class": QNetwork, "kwargs": { "memory_type": 'random'  } },
    ]
    for config in configs:
        with capture() as stdout:
            print(f'\nconfig: {config}')
            scores = []
            model_name = config['model_name']
            agent      = DQNPolicyAgent(state_size, action_size, model_class=config['model_class'], **config.get('kwargs',{}))
            # agent.load(f'models/{model_name}.pth')
            scores    += train_dqn(
                env,
                agent,
                max_t=1000,
                n_episodes=1000,
                filename=f'models/{model_name}.pth'
            )
            # agent.save(f'models/{model_name}.pth')

        print("\n".join(stdout))
        with open(f'models/{model_name}.log', 'w') as f:
            f.write("\n".join(stdout))

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.title(f'Plot of Rewards: {model_name}')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(f'models/{model_name}.png', bbox_inches='tight')
        plt.show()
