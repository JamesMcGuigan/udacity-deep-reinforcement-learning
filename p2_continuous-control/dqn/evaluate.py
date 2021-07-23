from unityagents import UnityEnvironment

from dqn.dqn_agent import DQNPolicyAgent


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


if __name__ == '__main__':
    env   = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    agent = DQNPolicyAgent.from_env(env)  #  state_size == 37, action_size == 4
    agent.load('model.pth')
    evaluate_agent(env, agent)
