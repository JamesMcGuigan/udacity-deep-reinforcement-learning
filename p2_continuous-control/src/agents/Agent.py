import os
from typing import List

import torch

from src.libs.Trajectory import Experience


class Agent(object):
    """ Base interface class for Unity agents """

    def __init__(self, state_size, action_size, num_agents, params={}):
        self.action_size = action_size
        self.state_size  = state_size
        self.num_agents  = num_agents
        self.params      = params
        self.actor       = self.get_actor_network()
        self.critic      = self.get_critic_network()


    def act(self, state, eps) -> int:
        pass

    def step(self, experience: Experience):
        (state, action, reward, next_state, done) = experience
        pass

    def get_actor_network(self):
        return None


    def get_critic_network(self):
        return None


    @property
    def persisted_fields(self) -> List[str]:
        """ These are the neural network fields to be persisted upon load/save """
        return ['actor', 'critic']


    def filename(self, name='', field=''):
        return f'./models/{self.__class__.__name__}.{name}.{field}.pth'.replace('..','.').replace('..','.')


    def load(self, name=''):
        for field in self.persisted_fields:
            filename = self.filename(field, name)
            try:
                if os.path.isfile(filename):
                    getattr(self, field).load_state_dict(torch.load(filename))
                    print(  f'{self.__class__.__name__}.load(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
                else: print(f'{self.__class__.__name__}.load(): {filename} path not found')
            except Exception as e:print(f'{self.__class__.__name__}.load(): {filename} exception: {e}')


    def save(self, name=''):
        for field in self.persisted_fields:
            filename = self.filename(field, name)
            try:
                torch.save(getattr(self, field).state_dict(), filename)
                print(f'\n{self.__class__.__name__}.save(): {filename} = {os.stat(filename).st_size/1024:.1f}kb')
            except Exception as e: print(f'{self.__class__.__name__}.save(): {filename} exception: {e}')
