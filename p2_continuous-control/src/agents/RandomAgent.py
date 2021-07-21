from typing import List

import numpy as np

from .Agent import Agent
from ..libs.Trajectory import Experience


class RandomAgent(Agent):

    def act(self, state, eps) -> np.ndarray:
        actions = np.random.randn(self.num_agents, self.action_size) # select an action (for each agent)
        return actions

    # Nothing to train here
    def step(self, experience: Experience):
        (state, action, reward, next_state, done) = experience
        pass

    # Nothing to save here
    @property
    def persisted_fields(self) -> List[str]: return []
    def load(self, name=''): pass
    def save(self, name=''): pass
