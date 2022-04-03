import os
import platform

import humanize
from unityagents import UnityEnvironment

from dqn_agent import DQNAgent
from model import QNetwork, DuelingQNetwork

def evaluate_agent(env, agent):
    brain_name  = env.brain_names[0]
    brain       = env.brains[brain_name]

    env_info  = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score
    while True:
        action     = agent.act(state)
        env_info   = env.step(action)[brain_name]       # send the action to the environment
        next_state = env_info.vector_observations[0]    # get the next state
        reward     = env_info.rewards[0]                # get the reward
        done       = env_info.local_done[0]             # see if episode has finished
        score     += reward                             # update the score
        state      = next_state                         # roll over the state to next time step
        if done:                                        # exit loop if episode finished
            break

    print("Score: {}".format(score))


def main(banana_path, config):
    env   = UnityEnvironment(banana_path)
    state_size, action_size = DQNAgent.get_env_state_action_size(env)
    agent = DQNAgent(state_size, action_size, model_class=config['model_class'], **config.get('kwargs',{}))
    agent.load(f"models/{config['model_name']}.pth")
    evaluate_agent(env, agent)


if __name__ == '__main__':
    if   platform.system() == 'Linux':  banana_path = "./Banana_Linux/Banana.x86_64"
    elif platform.system() == 'Darwin': banana_path = "./Banana.app"
    else: raise Exception('No Banana for OS')

    configs = [
        { "model_name": "dqn",           "model_class": QNetwork },
        # { "model_name": "dueling_dqn",   "model_class": DuelingQNetwork },
    ]
    for config in configs:
        print( 'UnityEnvironment:', banana_path, '[', humanize.naturalsize(os.path.getsize(banana_path)), ']', config )
        try:    main(banana_path=banana_path, config=config)
        except Exception as e: print('UnityEnvironment: Exception:', e)
