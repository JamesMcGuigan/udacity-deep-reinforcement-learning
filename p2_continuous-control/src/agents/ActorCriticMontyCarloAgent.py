from typing import List

import torch
from torch import optim, tensor

from src.agents.Agent import Agent
from src.libs.Trajectory import Trajectory
from src.libs.device import device
from src.networks.FCNet import FCNet
import torch.nn.functional as F


class ActorCriticMontyCarloAgent(Agent):
    """
    This implements Actor Critic Monty Carlo simulation

    BUG: First attempt, this doesn't converge'
    """

    def __init__(self,
                 state_size,
                 action_size,
                 num_agents,
                 params = {},
                 LR     = 1e-3,
                 GAMMA  = 0.99,  # discount factor
                 add_noise = False,
    ):
        super().__init__(state_size, action_size, num_agents, params)
        self.LR     = LR
        self.GAMMA  = GAMMA
        self.add_noise = add_noise

        self.actor  = FCNet(shape=(self.state_size, 32, 16, self.action_size))
        self.critic = FCNet(shape=(self.state_size, 32, 16, 1))


        self.optimizer_actor  = optim.Adam(self.actor.parameters(), lr=LR)
        self.optimizer_critic = optim.Adam(self.actor.parameters(), lr=LR)


    @property
    def persisted_fields(self) -> List[str]:
        """ These are the neural network fields to be persisted upon load/save """
        return ['actor']


    def act(self, state, eps=0) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state)
            if self.add_noise and eps > 0:
                noise    = eps * torch.empty(actions.shape).normal_(mean=0,std=0.25)
                actions += noise
                actions  = torch.clamp(actions, min=-1, max=1)

        self.actor.train()
        return actions.cpu().data.numpy()


    def step_trajectory(self, trajectory: Trajectory, eps=1.0):
        """ Agents may choose to define either step_experience() or step_trajectory() """

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        loss = self.loss(trajectory, eps)
        loss.backward()

        self.optimizer_actor.step()
        self.optimizer_critic.step()


    def loss(self, trajectory: Trajectory, eps=1.0):
        """
        δ ← R + γv(S′,w) − v(S,w)
        https://stats.stackexchange.com/questions/321234/actor-critic-loss-function-in-reinforcement-learning
        """
        states, actions, rewards, next_states, dones, idxs = zip(*trajectory.with_future_rewards(eps))
        # total_reward = sum(rewards)

        current_values = self.critic(states).squeeze(-1)
        next_values    = self.critic(next_states).squeeze(-1)
        # reward_values  = tensor([ total_reward for _ in range(len(states)) ]).float().to(device)
        reward_values  = tensor(rewards).squeeze(-1)

        critic_loss = F.mse_loss(reward_values, current_values)
        actor_loss  = reward_values + (self.GAMMA * next_values) - current_values  # QUESTION: should reward_values be negative?
        loss        = actor_loss.mean() + critic_loss.mean()
        return loss
