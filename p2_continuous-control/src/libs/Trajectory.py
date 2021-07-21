from collections import namedtuple, UserList
from typing import List

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class Trajectory(UserList):

    def with_future_rewards(self, eps=1.0) -> List[Experience]:
        trajectory    = list(self)
        future_reward = 0
        for n in range(len(self)):
            state, action, reward, next_state, done = trajectory[-n]
            future_reward *= eps
            future_reward += reward
            trajectory[-n] = Experience(state, action, future_reward, next_state, done)
        return trajectory


    def get_rewards(self) -> List[float]:
        states, actions, rewards, next_states, dones = zip(*self)
        return rewards


    def get_total_reward(self) -> float:
        states, actions, rewards, next_states, dones = zip(*self)
        total_reward = sum(rewards)
        return total_reward
