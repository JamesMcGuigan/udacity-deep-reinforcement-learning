# Changes for Continuous Actor Critic:
# - Policy head outputs μ mean and σ std
# - Policy loss is the negative log probability of a normal distribution
# - Entropy bonus uses the normal distribution equation
# https://youtu.be/kWHSH2HgbNQ?t=193
from torch import nn, is_tensor, tensor


def A2CNet():

    def __init__(self, state_size, action_size, hidden_size):
        self.base = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.ReLU(),  # [0,+]
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Tanh(), # [-1,1]
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softplus(), # [-0, +]
        )
        self.value = nn.Linear(hidden_size, 1)


    def forward(self, state):
        x     = (state if is_tensor(state) else tensor(state))
        x     = self.base(x)
        mu    = self.mu(x)
        var   = self.var(x)
        value = self.value(x)
        return mu, var, value
