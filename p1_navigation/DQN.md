# DQN

## Game Mechanics

The game here is Bananas, the agent exists in a room filled with both +1 yellow and -1 blue bananas. The aim is to collect as many +1 yellow bananas as possible within the episode time limit, and avoid collecting -1 blue bananas.

The agent can move: 
- 0 forward 
- 1 backwards
- 2 left 
- 3 right


> The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
> 
> **Ray Perception (35)**
> 
> 7 rays projecting from the agent at the following angles (and returned in this order):
>
> [20, 90, 160, 45, 135, 70, 110] # 90 is directly in front of the agent
> 
> **Ray (5)**
>
> Each ray is projected into the scene. If it encounters one of four detectable objects the value at that position in the array is set to 1. Finally there is a distance measure which is a fraction of the ray length.
> 
> [Banana, Wall, BadBanana, Agent, Distance]
> 
> example `[0, 1, 1, 0, 0.2]` There is a BadBanana detected 20% of the way along the ray and a wall behind it.
> 
> **Velocity of Agent (2)**
> 
> - Left/right velocity (usually near 0)
> - Forward/backward velocity (0-11.2)

- https://github.com/Unity-Technologies/ml-agents/issues/1134


## DQN Reinforcement Learning

The basic pattern of Reinforcement Learning assumes the agent exists in an environment, which at each timestep, 
offers up an observation and expects the agent to select an action in response. The environment also reports the
reward returned by each action and if the game has completed.

Agent decision-making process uses a DQN which is implemented as neural network, mapping the 37 observation dimensions
to 4 action dimensions, with the best action selected via argmax. 

DQN Reinforcement Learning to directly predict actions is [notoriously unstable](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf), but there are several Improvements to the algorithm that can improve things.



### Experience Replay

Dynamically generating a full game trajectory is expensive, and the amount learnt in each neural network epoch is small,
so performance can be optimized by caching previous trajectory game-states and sampling from it each time, thus reducing
the data generation costs.

Training with sequential order of events can lead to correlations between neighbouring datapoints which can result in
the neural network learning localized features. Sampling randomly from the dataset using experience replay removes this
correlation.


### Prioritized Experience Replay

Prioritized Experience Replay doesn't require the sampling to be random. Instead, the randomness can be weighted using
a heuristic such as event rarity or the temporal-difference error.


### Fixed Q-Targets

The DQN is trained based on the temporal-difference between the previous expected cumulative reward and
the actual reward experienced by the current trajectory replay.

However if the temporal difference is computed using its own output whilst being trained, the expected reward
can become a moving target, potentially creating feedback loops between training and exploration
that inhibits learning.

The solution to this is to use a cached version of the DQN for predicting the temporal-difference error, and
only update this periodically between epochs. This provides a fixed target for the neutral network to learn against.


### Double DQN

Single DQNs are known to overestimate action values under certain conditions. 

Double DQN trains two separate neural networks and uses each of them to provide the estimated Q values for the training
of the other network. Each DQN is trained on different sample data. This helps prevents overfitting.

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)

### Dueling DQN

DQNs often predict Q values where the difference between values is much smaller than the average magnitude of all values.
These difference in magnitude can result in noise when attempting to train the network.

A Dueling DQN splits the output into two heads, with two loss functions. 
One is a scalar score value for the current state. The other is the relative advantage for each action. 
These heads are then combined back to produce an aggregate Q value for each action. 

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### AC3 - Multi-Step Bootstrap Targets

Asynchronous training allows multiple independent agents to be trained in parallel. Each training from a copy of the 
master neural network parameters, often in minibatches. Periodically the differences in child networks weights will 
be reapplied to the master network, and the updated values from the master network copied back to the child.

This allows for training of large models to happen over multiple machines.

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)


### Distributional DQN

Instead of predicting a single value for each action, a Distributional DQN predicts the probability distribution of
potential rewards for each action (using bins for different percentiles). This makes the distinction that some 
actions may have reliable outcomes, whilst others may have risky rewards. 

This slightly increases the computational cost of the network, but can result in reduction in training steps required
to achieve superior performance.

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Learning Distributions over Rewards leads to State-of-the-art in RL](https://towardsdatascience.com/learning-distributions-over-rewards-leads-to-state-of-the-art-in-rl-5afbf70672e)


### Noisy DQN

A NoisyNet DQN modifies the neural network weights by adding a tensor of random noise to the linear layers. 
This noise vector can be changed after each epoch.

This can induce a consistent, and potentially very complex, state-dependent change in policy over multiple time steps, 
for the purposes of aiding exploration of the state space.

- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- [NoisyNet pytorch implementation](https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py)


### Rainbow DQN

Combining all the above improvements into the DQN algorithm is known as as Rainbow DQN, and can improve performance
beyond that obtained using these improvements indivudally.


### Multi Frame Input

To detect velocity and motion within otherwise static pixel data, it is possible to batch the input into
2-4 timeframe slices, which will expose the neutral network to relative changes over time. The same action
can be persisted until the end of the next batch window.

sa


# Papers

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

- [Issues in Using Function Approximation for Reinforcement Learning - Sebastian Thrun, Anton Schwartz](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf)

- [Human-level control through deep reinforcement
  learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

- [Improvements in Deep Q Learning: Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets](https://medium.com/free-code-camp/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)

- [How to Read and Understand a Scientific Paper: A Step-by-Step Guide for Non-Scientists](https://www.huffpost.com/entry/how-to-read-and-understand-a-scientific-paper_b_5501628)
