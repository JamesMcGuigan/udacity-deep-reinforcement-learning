import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class QPolicyNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, granularity, seed, fc1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.granularity = granularity

        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size * granularity)

    def forward(self, state):
        """
        Build a network that maps state -> action values
        Continuous policy version has self.granularity buckets, which are mapped to [-1,1] range
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_probs  = x.reshape(x.shape[0], self.action_size, self.granularity)  # range [0,1]
        action_cats   = Categorical(F.relu(action_probs)).sample()     # range [0,20]
        action_values = (action_cats / self.granularity * 2) - 1       # range [-1,1]
        return action_values



# Inspired by: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=16, fc3_units=8):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.features = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size),
        )


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x          = state
        x          = self.features(x)
        values     = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals      = values + (advantages - advantages.mean())
        return qvals
