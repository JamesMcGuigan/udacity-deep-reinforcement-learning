import os
import random
from collections import namedtuple, deque
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dqn.ReplayBuffer import ReplayBuffer
from dqn.model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, model_class=QNetwork, update_type='dqn', seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        random.seed(seed)

        self.state_size  = state_size
        self.action_size = action_size
        self.update_type = update_type

        # Q-Network
        self.qnetwork_local  = model_class(state_size, action_size, seed).to(device)
        self.qnetwork_target = model_class(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # try:                   qnetwork_parameters = self.qnetwork_local.parameters()
        #     except Exception as e: qnetwork_parameters = {}; print('self.qnetwork_local.parameters()', e)
        #     self.optimizer = optim.Adam(qnetwork_parameters, lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    ### Load / Save Functions ###

    @classmethod
    def get_env_state_action_size(cls, env) -> Tuple[int, int]:
        # env         = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
        brain_name  = env.brain_names[0]
        brain       = env.brains[brain_name]
        env_info    = env.reset(train_mode=True)[brain_name]
        state       = env_info.vector_observations[0]

        state_size  = state.shape[0]                  # state.shape == (37,)
        action_size = brain.vector_action_space_size  # action_size == 4
        return state_size, action_size


    def load(self, filename):
        try:
            filename = os.path.abspath(filename)
            if os.path.isfile(filename):
                self.qnetwork_local.load_state_dict(torch.load(filename))
                print(f'{self.__class__.__name__}.load(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
            else:              print(f'{self.__class__.__name__}.load(): {filename} path not found')
        except Exception as e: print(f'{self.__class__.__name__}.load(): {filename} exception: {e}')


    def save(self, filename):
        filename = os.path.abspath(filename)
        try:
            torch.save(self.qnetwork_local.state_dict(), filename)
            print(f'\n{self.__class__.__name__}.save(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
        except Exception as e: print(f'{self.__class__.__name__}.save(): {filename} exception: {e}')



    ### Agent Interaction Functions ###

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss           = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_( tau*local_param.data + (1.0-tau)*target_param.data )
