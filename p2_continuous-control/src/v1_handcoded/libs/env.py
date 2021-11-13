from typing import Tuple


def get_env_state_action_agents_size(env) -> Tuple[int, int, int]:
    brain_name  = env.brain_names[0]
    brain       = env.brains[brain_name]
    env_info    = env.reset(train_mode=True)[brain_name]
    state       = env_info.vector_observations[0]

    state_size  = state.shape[0]                  # state.shape == (37,)
    action_size = brain.vector_action_space_size  # action_size == 4
    num_agents  = len(env_info.agents)
    return state_size, action_size, num_agents
