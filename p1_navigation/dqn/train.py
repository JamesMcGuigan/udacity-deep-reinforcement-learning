from collections import deque, defaultdict

import numpy as np
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from dqn.ReplayBuffer import Experience
from dqn.dqn_agent import DQNAgent
from dqn.model import QNetwork


def train_dqn(
        env, agent, filename,
        obs_attr='vector_observations',  # vector_observations || visual_observations
        # obs_attr='visual_observations',  # vector_observations || visual_observations
        n_episodes=2000,
        max_t=100,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.99,
        score_window_size=100, win_score=13,
        exit_after_first_reward=False,
        future_reward_decay=0,
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
            'future_reward_decay': future_reward_decay,
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
                action     = agent.act(state, eps)
                env_info   = env.step(action)[brain_name]       # send the action to the environment
                next_state = getattr(env_info, obs_attr)[0]
                reward     = env_info.rewards[0]                # get the reward
                done       = env_info.local_done[0]             # see if episode has finished

                trajectory.append( Experience(state, action, reward, next_state, done) )

                state      = next_state                         # roll over the state to next time step
                score     += reward

                if exit_after_first_reward and score != 0: break  # keep going until at least one yellow banana
                if done: break

            # As rewards are sparse, using decayed future reward allows training on events leading upto a banana
            # This tends to speed up very early stage training, but tends to converge to lower average scores
            #   future_reward_decay = 1.0 | Episode 2000	Average Score: 4.33
            #   future_reward_decay = 0.9 | Episode 2000	Average Score: 3.56
            #   future_reward_decay = 0.0 | Episode 2000	Average Score: 5.51
            if future_reward_decay:
                future_reward = 0
                for n in range(len(trajectory)):
                    state, action, reward, next_state, done = trajectory[-n]
                    future_reward *= future_reward_decay
                    future_reward += reward
                    trajectory[-n] = Experience(state, action, future_reward, next_state, done)

            for experience in trajectory:
                state, action, reward, next_state, done = experience
                agent.step(state, action, reward, next_state, done)


            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {:4d}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {:4d}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= win_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                      .format(i_episode-100, np.mean(scores_window)))
                agent.save(filename)

    except KeyboardInterrupt as exception:
        # agent.save(filename)
        pass

    return scores



if __name__ == '__main__':
    env   = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    state_size, action_size = DQNAgent.get_env_state_action_size(env)  #  state_size == 37, action_size == 4

    scores = defaultdict(list)
    for future_reward_decay in [1, 0.9, 0]:
        modelname = f'dqn@future_reward_{future_reward_decay}'
        agent     = DQNAgent(state_size, action_size, model_class=QNetwork, update_type='dqn')
        scores[future_reward_decay] = train_dqn(
            env,
            agent,
            n_episodes=2000,
            future_reward_decay=future_reward_decay,
            filename=f'models/{modelname}.pth'
        )

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores[future_reward_decay])), scores[future_reward_decay])
        plt.title('Plot of Rewards')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(f'models/{modelname}.png', bbox_inches='tight')
        plt.show()


    for future_reward_decay, score in scores.items():
        print(f'eps_decay = {future_reward_decay} | average_score = {np.mean(score[:-100])}')
