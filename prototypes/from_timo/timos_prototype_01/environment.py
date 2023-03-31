from typing import Dict, Tuple, List
from src.custom_types import Behavior, MTDTechnique, actions, supervisor_map
import numpy as np
import random


# handles the supervised, online-simulation of episodes
class SensorEnvironment:

    def __init__(self, train_data: Dict[Behavior, np.ndarray] = None):
        self.train_data = train_data
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.train_data[Behavior.RANSOMWARE_POC][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]

    # Returns a randomly selected attack state with non normal behaviour.
    def sample_random_attack_state(self):
        """i.e. for starting state of an episode,
        (with replacement; it is possible that the same sample is chosen multiple times)"""
        rb = random.choice([b for b in Behavior if b != Behavior.NORMAL])
        attack_data = self.train_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=1), :]

    # Return random sample with specified behaviour
    def sample_behavior(self, b: Behavior):
        behavior_data = self.train_data[b]
        return behavior_data[np.random.randint(behavior_data.shape[0], size=1), :]

    def step(self, action: int):
        current_behavior = self.current_state.squeeze()[-1]

        if current_behavior in supervisor_map[action]:
            # print("correct mtd chosen according to supervisor")
            new_state = self.sample_behavior(Behavior.NORMAL)
            reward = self.calculate_reward(True)
            isTerminalState = True
        else:
            # print("incorrect mtd chosen according to supervisor")
            new_state = self.sample_behavior(current_behavior)
            reward = self.calculate_reward(False)
            isTerminalState = False

        self.current_state = new_state
        return new_state, reward, isTerminalState

    def reset(self):
        self.current_state = self.sample_random_attack_state()
        return self.current_state

    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    def calculate_reward(self, success):
        """
        this method can be refined to distinguish particularly wasteful MTDs (i.e. Dirtrap penalized harder than rootkit sanitization)
        """
        if success:
            return 1
        else:
            return -1
