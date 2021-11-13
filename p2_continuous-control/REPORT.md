# Project 2: Continuous Control

See Also:
- [INSTALL.md](INSTALL.md)
- [README.md](README.md)
- [RUBRIC.md](RUBRIC.md)


## Random Agent
```
$ python3 src/v1_handcoded/test_random_bot.py
```

This is a baseline agent mostly for the purposes of testing the training loop code.

It typically returns a score of zero, but occasionally scores as high as 0.9. 
Performance is random and doesn't improve over time.

![](models/RandomAgent.png)


## Actor Critic
```
$ python3 src/v1_handcoded/train.py
```

An attempt was made to handcode implementations of the Actor Critic algorithm.

This code was able to successfully loop, but training failed to converge and this 
achieved similar performance to the RandomBot.

- [src/v1_handcoded/agents/ActorCriticMontyCarloAgent.py](src/v1_handcoded/agents/ActorCriticMontyCarloAgent.py)
- [src/v1_handcoded/agents/AgentA2C.py](src/v1_handcoded/agents/AgentA2C.py)


## Stable Baselines
```
$ python3 src/v3_stable_baselines/example_unity.py
```

An alternative to writing (potentially buggy) implementations of the algorithms is to use 
a library implementation. Stable Baselines was chosen as a framework and a wrapper class was
written to enable a Unity environment to emulate the OpenAI Gym interface.

Strangely, this code was able to successfully loop, but training failed to converge and this
achieved similar performance to both RandomBot and hand-coded Actor Critic implementations.


## DQN Policy Agent
```
$ python3 src/v2_dqn/train.py
```

An attempt was made to port the DQN code from the Banana game with a discrete action spaces 
and map it to using a bucketed policy distribution, with the range [-1,+1] 
represented as a size 20 vector allowing for 0.1 granularity in motor control.

There is yet to be solved bug in the pytorch implementation, with the new data structures
resulting in a RuntimeError that `loss.backward()` does not have a grad_fn()

- [dqn/v2_dqn/dqn_agent.py](src/v2_dqn/dqn_agent.py)
- [dqn/v2_dqn/model.py](src/v2_dqn/model.py)
