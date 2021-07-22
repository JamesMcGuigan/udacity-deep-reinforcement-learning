# NOTE: training using SumTreeReplayBuffer fails to converge

# Source: https://raw.githubusercontent.com/rlcode/per/master/SumTree.py
import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
import torch

from src.dqn.ReplayBuffer import Experience, device, ReplayBuffer


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        ### BUGFIX: Maximum recursion depth exceeded for idx == 99,999- James McGuigan
        # parent = (idx - 1) // 2
        # self.tree[parent] += change
        # if parent != 0:
        #     self._propagate(parent, change)

        # BUGFIX: unroll recursion - James McGuigan
        while True:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            if   parent  > 0: idx = parent
            elif parent <= 0: break

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


# Source: https://github.com/rlcode/per/blob/master/prioritized_memory.py

import random
import numpy as np
# from SumTree import SumTree

class SumTreeMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)



### Shared Interface with ReplayBuffer

class SumTreeReplayBuffer(SumTreeMemory):  # stored as ( s, a, r, s_ ) in SumTree
    """
    Wrapper class around SumTreeMemory to provide a common interface with ReplayBuffer
    @Author James McGuigan
    """

    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        self.action_size   = action_size  # unused
        self.batch_size    = batch_size
        self.buffer_size   = buffer_size
        self.seed          = seed         # unused
        super(SumTreeReplayBuffer, self).__init__(buffer_size)

    def __len__(self):
        return self.tree.n_entries

    def sample(self):
        # SumTreeMemory may return 0 for samples that have not been initialized, so filter them out
        experiences = []
        idxs        = []
        _experiences, _idxs, is_weight = super(SumTreeReplayBuffer, self).sample(self.batch_size)
        for experience, idx in zip(_experiences, _idxs):
            if isinstance(experience, tuple):
                experiences.append(experience)
                idxs.append(idx)
                if len(idxs) == self.batch_size: break
        if len(experiences) == 0:
            return None

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences if e is not None])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences if e is not None])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        idxs        = torch.from_numpy(np.vstack([idx          for idx in idxs      if idx is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, idxs)


    def add(self, state, action, reward, next_state, done, idx, error=0):
        # QUESTION: should error be initialized to 0?
        sample = Experience(state, action, reward, next_state, done, idx)
        super(SumTreeReplayBuffer, self).add(error, sample)

    def update(self, experiences, td_errors):
        states, actions, rewards, next_states, dones, idxs = experiences
        assert len(idxs) == len(td_errors)
        for i in range(len(idxs)):
            idx      = int(idxs[i])
            td_error = float(td_errors[i])
            super(SumTreeReplayBuffer, self).update(idx, td_error)
