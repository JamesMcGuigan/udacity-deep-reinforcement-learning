import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32, fc3_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units,  fc2_units)
        self.fc3  = nn.Linear(fc2_units,  fc3_units)
        self.fc4  = nn.Linear(fc3_units,  action_size)

        self.bn1  = nn.BatchNorm1d(num_features=fc1_units)
        self.bn2  = nn.BatchNorm1d(num_features=fc2_units)
        self.bn3  = nn.BatchNorm1d(num_features=fc3_units)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.softmax(self.fc4(x))

        return x
