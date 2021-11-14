#!/usr/bin/env python
# This is the main training loop

import os

from unityagents import UnityEnvironment

from src.v1_handcoded.libs.contextmanager import capture
from src.v1_handcoded.libs.env import get_env_state_action_agents_size
from src.v4_ddpg_bipedal.ddpg_agent import DDPGAgent

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))  # ensure cwd in project root
    # env = UnityEnvironment(file_name='./unity/Reacher_Linux_NoVis/Reacher.x86_64')
    env = UnityEnvironment(file_name='./unity/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
    # env = UnityEnvironment(file_name='./unity/Crawler_Linux_NoVis/Crawler.x86_64')
    state_size, action_size, num_agents = get_env_state_action_agents_size(env)  # == (33, 4, 20)

    configs = [
        { "model_name": "", "agent_class": DDPGAgent, "params": {}},
    ]
    for config in configs:
        with capture() as stdout:
            print(f'\nconfig: {config}')
            rewards = []

            agent_class = config['agent_class']
            agent = agent_class(
                state_size  = state_size,
                action_size = action_size,
                # num_agents  = num_agents,
                # params      = config.get('params',{}),
                # **config.get('kwargs',{})
            )
            model_name  = config['model_name'] or agent.__class__.__name__

            # agent.load()
            rewards += agent.train(env, n_episodes=20_000)
            # agent.save()

        print("".join(stdout))

        filename = agent.filename(name=model_name, ext='.log')
        with open(filename, 'w') as f:
            f.write("".join(stdout))

        agent.plot_rewards(rewards, model_name)

    env.close()
