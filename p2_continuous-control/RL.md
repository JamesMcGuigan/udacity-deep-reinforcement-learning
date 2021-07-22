# Reinforcement Learning

### Comparison of reinforcement learning algorithms

| Algorithm | Description | Policy | Action Space | State Space | Operator |
|-----------|-------------|--------|--------------|-------------|----------|
| Monte Carlo | Every visit to Monte Carlo | Either | Discrete | Discrete | Sample-means |
| Q-learning | State–action–reward–state | Off-policy | Discrete | Discrete | Q-value |
| SARSA | State–action–reward–state–action | On-policy | Discrete | Discrete | Q-value |
| Q-learning - Lambda | State–action–reward–state with eligibility traces | Off-policy | Discrete | Discrete | Q-value |
| SARSA - Lambda | State–action–reward–state–action with eligibility traces | On-policy | Discrete | Discrete | Q-value |
| DQN | Deep Q Network | Off-policy | Discrete | Continuous | Q-value |
| DDPG | Deep Deterministic Policy Gradient | Off-policy | Continuous | Continuous | Q-value |
| A3C | Asynchronous Advantage Actor-Critic Algorithm | On-policy | Continuous | Continuous | Advantage |
| NAF | Q-Learning with Normalized Advantage Functions | Off-policy | Continuous | Continuous | Advantage |
| TRPO | Trust Region Policy Optimization | On-policy | Continuous | Continuous | Advantage |
| PPO | Proximal Policy Optimization | On-policy | Continuous | Continuous | Advantage |
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | Off-policy | Continuous | Continuous | Q-value |
| SAC | Soft Actor-Critic | Off-policy | Continuous | Continuous | Advantage |

- https://en.wikipedia.org/wiki/Reinforcement_learning#Comparison_of_reinforcement_learning_algorithms



RL Paper Reading 20-Summary of MF Algorithms (VPG, TROP, PPO, DDPG, TD3, SAC)
- https://www.programmersought.com/article/96325908734/
