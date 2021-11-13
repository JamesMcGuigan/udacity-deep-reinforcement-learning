import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


class DummyBody(nn.Module):
    def __init__(self):
        super(DummyBody, self).__init__()

    def forward(self, x):
        return x


class FCBody(nn.Module):
    def __init__(self, state_size, output_size, hidden_size, gate=DummyBody()):
        super(FCBody, self).__init__()

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.gate = gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.gate(self.linear3(x))

        return x


class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCriticNet, self).__init__()

        self.actor_body  = FCBody(state_size, action_size, hidden_size, F.tanh)
        self.critic_body = FCBody(state_size, 1, hidden_size)
        self.std         = nn.Parameter(torch.ones(1, action_size))

    def forward(self, obs, action=None):
        obs = tensor(obs)
        a = self.actor_body(obs)
        v = self.critic_body(obs)

        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v
