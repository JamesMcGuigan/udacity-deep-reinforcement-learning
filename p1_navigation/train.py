from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from dqn_agent import Agent


def train_dqn(
    env, agent, filename,
    obs_attr='vector_observations',  # vector_observations || visual_observations
    n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
    score_window_size=100, win_score=13,
    exit_after_first_reward=False,
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
    scores = []                        # list containing scores from each episode
    try:
        print('train_dqn(', {
            'env': env,
            'agent': agent,
            'filename': filename,
            'obs_attr': obs_attr,
            'n_episodes': n_episodes,
            'max_t': max_t,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'score_window_size': score_window_size,
            'win_score': win_score,
            'exit_after_first_reward': exit_after_first_reward,
        }, ')')
        
        brain_name  = env.brain_names[0]
        brain       = env.brains[brain_name]
        action_size = brain.vector_action_space_size        # action_size == 4

        scores_window = deque(maxlen=score_window_size)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state    = getattr(env_info, obs_attr)[0]    # get the next state

            score = 0
            for t in range(max_t):
                action     = agent.act(state)
                env_info   = env.step(action)[brain_name]       # send the action to the environment
                next_state = getattr(env_info, obs_attr)[0]
                reward     = env_info.rewards[0]                # get the reward
                done       = env_info.local_done[0]             # see if episode has finished
                state      = next_state                         # roll over the state to next time step
                score     += reward

                if exit_after_first_reward and score != 0: break
                if done: break

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= win_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                agent.save(filename)

    except KeyboardInterrupt as exception:
        agent.save(filename)

    return scores



if __name__ == '__main__':
    env   = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    agent = Agent.from_env(env)  #  state_size == 37, action_size == 4
    agent.load('checkpoint.pth')

    # Increment max_t to assist with finding the first banana
    scores  = []
    scores += train_dqn(env, agent, filename='checkpoint.pth', exit_after_first_reward=True)
    scores += train_dqn(env, agent, filename='checkpoint.pth')
    agent.save('checkpoint.pth')

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
