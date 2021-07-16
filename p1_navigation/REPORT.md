# Project 1: Navigation with Deep Q Learning

## Evaluation 
[src/evaluate.py](src/evaluate.py)

```
conda activate drlnd
~/.anaconda3/envs/drlnd/bin/python src/evaluate.py
```

This shows the basic agent-environment interaction loop using a pretrained model


## Training
[src/train.py](src/train.py)

```
conda activate drlnd
~/.anaconda3/envs/drlnd/bin/python src/train.py
```

Training simulates upto 5000 episodes with 1000 timesteps each. 
- For each episode a game state trajectory is recorded and added to the ReplayBuffer
- If `use_future_reward=True` then epsilon decayed future rewards for the trajectory are added 
    - Using future rewards improves early training speed, but doesn't seem to increase the maximum score
  to experience training data, but training is delayed until after the episode has completed
- The neural network is trained with a minibatch of 64 experiences every 4 timesteps

### DQN Agent 
[src/dqn_agent.py](src/dqn_agent.py)

- Uses epsilon-greedy action selection, which decays throughout the training from 1 ^ 0.995 for each episode
- Optionally implements a Dueling DQN architecture
- Dueling DQN with 1000 timesteps reaches average score of 13 slightly in less episodes (374 vs 436) but requires
  the same amount of compute power


### Basic DQN Model Architecture

[src/model.py](src/model.py)

- This implements a simple 3-layer fully connected network, sufficient to implement all logical gates
- Network size was originally chosen by guess with size 64x64x4 (roughly double the input size of 37)
- Brute force search over different configurations showed:
  - 8x8   is almost capable of solving the banana navigation game, but in twice the amount of time
  - 16x8  is the smallest feasible network 
  - 32x16 has the fastest training time and requires the least number of episodes (selected as new default)
  - 64x64 was a little oversized (better than undersized), but not unreasonable as a blind guess
  - triangular networks are more performant than square networks

```
 8x8  - Environment almost in 1000 episodes! Average Score: 12.75 | Time: 882.1s
16x8  - Environment solved in  472 episodes! Average Score: 13.04 | Time: 542.5s
16x16 - Environment solved in  487 episodes! Average Score: 13.07 | Time: 539.3s
32x16 - Environment solved in  383 episodes! Average Score: 13.04 | Time: 450.3s
32x32 - Environment solved in  385 episodes! Average Score: 13.02 | Time: 459.1s
64x32 - Environment solved in  377 episodes! Average Score: 13.04 | Time: 455.0s
64x64 - Environment solved in  422 episodes! Average Score: 13.05 | Time: 487.1s
```

### Dueling DQN Model Architecture
[src/model.py](src/model.py)

- This implements a fully connected network with two heads and a recombination function.
- Network sizes guessed based on optimal values for Basic DQN Model Architecture (32x16 + 16x1 + 16x4)
- Shared Feature segment is a 2-layer fully-connected network copied from the Basic DQN
- Value and Advantage heads are also 2-layer fully-connected networks (with 16x16 square interface with Features)
- Outputs: Value + (Advanage - mean(Advanage)) 


### Prioritized Experience Replay
[src/SumTree.py](src/SumTree.py)

An alternative to random sampling is to weight using the temporal-difference error. 

In theory this should speed up training performance by prioritizing samples that differ most from expectations. 

This was implemented using the SumTree library, with a wrapper to allow it to share the same interface as the
existing ReplayBuffer. There is potentially a bug in this code, or an incorrect formula for the TD error.
Training using this memory structure fails to converge.


# Hyperparameters

In order of presumed importance:
- LR = 1e-3               # optimized learning rate
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- BUFFER_SIZE = int(1e5)  # 88 Mb memory + 100 previous games  
- BATCH_SIZE = 64         # sensible default
- UPDATE_EVERY = 4        # sensible default


## Learning Rate

LR Learning Rate is often the most important hyperparameter. 
- 1e-2 is too large and prevents the model from converging to optimum score
- 1e-3 seems to be the optimal LR value
- 1e-4 and 1e-5 are too small and increases training time by double or more

```
lr=1e-2 Environment unsolved after 1000 episodes!  Average Score:  8.57 | Time 934s
lr=1e-3 Environment solved in 457 episodes!        Average Score: 13.02 | Time 536s
lr=1e-4 Environment solved in 674 episodes!        Average Score: 13.04 | Time 781s
lr=1e-5 Environment unsolved after 1000 episodes!  Average Score: 11.80 | Time 962s 
```

## Gamma

Gamma is the discount factor for the Target Q Network when comparing it to actual rewards as part of the loss function.

percentage when soft updating between local and target Q neural networks.

Hyperparameter tuning shows 0.9-1.0 are reasonable values, though 0.99 is optimal. Smaller values fail to converge.

```
models/dqn@gamma=1.log    | Environment solved in 494 episodes!         Average Score: 13.06 | Time: 553s
models/dqn@gamma=0.99.log | Environment solved in 430 episodes!         Average Score: 13.03 | Time: 485s
models/dqn@gamma=0.9.log  | Environment solved in 449 episodes!         Average Score: 13.00 | Time: 506s
models/dqn@gamma=0.5.log  | Environment unsolved after 1000 episodes!   Average Score:  1.49 | Time: 929s
models/dqn@gamma=0.log    | Environment unsolved after 1000 episodes!   Average Score:  0.43 | Time: 938s
```

## Tau

Tau is the percentage when soft updating between local and target Q neural networks.

Hyperparameter tuning shows 1e-3 = 0.001 is the optimal value

```
models/dqn@tau=1e-1.log | Environment solved in 704 episodes!          Average Score: 13.00 | Time: 744s
models/dqn@tau=1e-2.log | Environment solved in 535 episodes!          Average Score: 13.02 | Time: 579s
models/dqn@tau=1e-3.log | Environment solved in 401 episodes!          Average Score: 13.01 | Time: 517s
models/dqn@tau=1e-4.log | Environment unsolved after 1000 episodes!    Average Score:  8.81 | Time: 962s
models/dqn@tau=1e-5.log | Environment unsolved after 1000 episodes!    Average Score:  0.19 | Time: 943s
```

## Buffer Size

Each Experience tuple requires 88 bytes. Each trajectory is 1000 time steps. 

BUFFER_SIZE = 1e5 = 100,000 * 88 bytes = 88 Megabytes of memory
BUFFER_SIZE = 1e5 = 100,000 / 1000     = 100 previous games

Thus 1e5 seems like a sensible guess for this value.


## Batch Size

64 is a sensible default for that will be left unoptimized.

This is small enough to fit into GPU memory, and large enough to make use of GPU parallelism, 
without sitting at the extreems of 1 = Stochastic Gradient Descent and N = Full-Batch Gradient Descent.

BATCH_SIZE=64 also interacts with UPDATE_EVERY=4 + BUFFER_SIZE=100_000. On average, 64 / 4 = 16
datapoints will be trained for every new datapoint added to the memory buffer. Once the buffer is full this means
each datapoint will be reused on average 16 times, the probability of which is independent of the buffer size.


# Plot of Rewards

If `max_t` is limited to only `100` timesteps, then the plot of rewards shows the network starts 
with an average score of 0 and eventually converges to an average score of around 5 around 1500 episodes, 
with similar results for both basic DQN and Dueling DQN networks and training for extended periods of time
does not seem to improve this.

![](models/dqn@future_reward_1.png)

To achieve a score of 13+ requires increasing `max_t` to a higher number such as `1000`, to allow the agent time to
collect additional bananas.

![](models/dqn.png)

![](models/dueling_dqn.png)


# Ideas for Future Work

Main difficulty at the moment is getting the DQN to converge towards
an increasing score. 

If this basic element could be achieved, then it might be worth attempting to optimize this score using the various components of a Rainbow DQN:
- Dueling Network Architecture (DONE)
- Double DQN 
- Prioritised Experience Replay
- Multi-step Returns
- Distributional RL 
- Noisy Nets
