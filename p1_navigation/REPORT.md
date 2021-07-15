# Project 1: Navigation with Deep Q Learning

## [evaluate.py](src/evaluate.py)

```
conda activate drlnd
python3 dqn/evaluate.py
```

This shows the basic agent-environment interaction loop using a pretrained model: 

```
while not done:
   action = agent.act(state)
   state, reward, done  = env.step(action)
```

- `agent.act()` reads in the agent state and predicts the next action (using a Deep Q Network)
- `env` is the a UnityEnvironment representing the game, which the agent must navigate through


## [train.py](src/train.py)

```
conda activate drlnd
python3 dqn/train.py
```

Training follows a similar code agent-environment interaction loop, but with a few modifications:
- Training happens inside a doubly nested loop, simulating multiple games run until a fixed timestep. 
- `agent.step(state, action, reward, next_state, done)` is called which adds the last experience to the ReplayBuffer and every 4 turns runs one epoch of neural network training using a random subsample of experience.

Issues:
- The code is mostly ported (with modifications) from the dqn exercise notebooks.
- The training loop itself executes
- The deep Q neural network doesn't seem to converge to a winning strategy
- Even after 2000 epochs, the average score tends towards zero with any clear indication of actual learning
- Sometimes the agent learns to run backwards (towards random bananas it can't see)
- This suggests an important step is missing from the training loop, or else the loss function or neural network shape is incorrect


## [dqn_agent.py](src/dqn_agent.py)

This mostly mirrors the code from the DQN example notebooks, but adds in `.load()`, `.save()` and `.from_env()` helper methods


## [model.py](src/model.py)

This represents the deep Q neural network behind the agent.

To keep things simple we are using a fully connected 64/32/16/4 triangular network. This is slightly larger than the input state and the shape and depth allows for building boolean logic gates inside the network.

Adding in BatchNorm then Softmax to the last layer doesn't seem to have any additional effect


# Plot of Rewards

The plot of rewards shows a random walk around the zero score, without ever converging on a winning strategy.

![](models/dqn@future_reward_1.png)

# Ideas for Future Work

Main difficulty at the moment is getting the DQN to converge towards
an increasing score. 

If this basic element could be achieved, then it might be worth attempting to optimize this score using the various components of a Rainbow DQN:
- DQN 
- Double DQN 
- Prioritised Experience Replay
- Dueling Network Architecture 
- Multi-step Returns
- Distributional RL 
- Noisy Nets

Another idea inspired from the work on the Atari games is to batch the environment input into 4 steps per decision point. This might provide additional information regarding the relative velocities of different objects  
