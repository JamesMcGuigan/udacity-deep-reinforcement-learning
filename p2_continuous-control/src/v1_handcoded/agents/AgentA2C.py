"""
A2C inspired by: https://github.com/colinskow/move37/blob/master/actor_critic/a2c_continuous.py

BUG: Fails to converge
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, optim

from src.v1_handcoded.agents.Agent import Agent
from src.v1_handcoded.libs.Trajectory import Trajectory
from src.v1_handcoded.libs.device import device


class ModelA2C(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        ).to(device)

        self.mu = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        ).to(device)

        self.var = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softplus(),
        ).to(device)

        self.value = nn.Linear(hidden_size, 1).to(device)


    def forward(self, x):
        x        = (x if torch.is_tensor(x) else tensor(x)).float().to(device)
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)



class AgentA2C(Agent):
    def __init__(
            self,
            state_size,
            action_size,
            num_agents,
            params = {},
            LR     = 1e-3,
            ENTROPY_BETA = 1e-4,
    ):
        super().__init__(state_size, action_size, num_agents, params)
        self.network   = ModelA2C(state_size, action_size).to(device)  # TODO: implement num_agents
        self.LR        = LR
        self.ENTROPY_BETA = ENTROPY_BETA

        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)


    def __call__(self, state, eps=0):
        return self.act(state)


    def act(self, state, eps=0):
        state = state if is_tensor(state) else tensor(state)
        state = state.float().to(device)

        mu_v, var_v, value_v = self.network(state)
        mu      = mu_v.data.cpu().numpy()
        sigma   = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions


    def step_trajectory(self, trajectory: Trajectory, eps=1.0):
        """ Agents may choose to define either step_experience() or step_trajectory() """
        loss = self.loss(trajectory, eps)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def loss(self, trajectory, eps=0):
        losses = torch.zeros((len(trajectory), self.action_size))
        for n, experience in enumerate(trajectory.with_future_rewards(eps)):
            (state, action, reward, next_state, done, idx) = experience
            action = tensor(action).to(device)
            reward = tensor(reward).to(device)

            mu_v, var_v, value_v = self.network(state)
            # loss_value_v   = F.mse_loss(value_v.squeeze(-1), reward)
            loss_value_v   = F.mse_loss(value_v, reward)
            adv_v          = reward.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v     = adv_v * self.calc_logprob(mu_v, var_v, action)
            loss_policy_v  = -log_prob_v.mean()
            entropy_loss_v = self.ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()
            loss_v         = loss_policy_v + entropy_loss_v + loss_value_v
            losses[n]      = loss_v

        loss = torch.mean(losses)
        return loss


    def calc_logprob(self, mu_v, var_v, actions_v):
        # with torch.no_grad():
        actions_v = tensor(actions_v)
        p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2
