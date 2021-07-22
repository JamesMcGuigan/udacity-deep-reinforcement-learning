import torch
from torch import nn, tensor
import torch.nn.functional as F

from src.libs.device import device


class FCNet(nn.Module):
    """
    This represents a simple fully connected network
    """
    def __init__(self, shape=(32,16,8), gate=torch.tanh, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # print([ (size1, size2) for size1, size2 in zip(shape[:-1], shape[1:]) ])
        self.linears = nn.ModuleList([
            nn.Linear(size1, size2)
            for size1, size2 in zip(shape[:-1], shape[1:])
        ])
        self.gate = gate


    def forward(self, state):
        """ Build a network that maps state -> action values """
        x = state if torch.is_tensor(state) else tensor(state)
        x = x.float().to(device)
        for n in range(len(self.linears)):
            x = self.linears[n](x)
            if n != len(self.linears)-1:
                x = F.relu(x)
        x = self.gate(x)
        return x
