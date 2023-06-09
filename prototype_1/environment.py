from typing import Dict, Tuple, List
from collections import defaultdict
from custom_types import Behavior, MTDTechnique
from scipy import stats
from tabulate import tabulate
import numpy as np
import pandas as pd
import os
import random
from data_manager import DataManager

# define MTD - (target Attack) Mapping
# indices corresponding to sequence
actions = (MTDTechnique.CNC_IP_SHUFFLE, MTDTechnique.ROOTKIT_SANITIZER,
           MTDTechnique.RANSOMWARE_DIRTRAP, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE)

supervisor_map: Dict[int, Tuple[Behavior]] = defaultdict(lambda: -1, {
    # MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    0: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK),
    1: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK),
    2: (Behavior.RANSOMWARE_POC,),
    3: (Behavior.RANSOMWARE_POC,)
})


# handles the supervised, online-simulation of episodes
class SensorEnvironment:

    def __init__(self, train_data: Dict[Behavior, np.ndarray] = None, test_data: Dict[Behavior, np.ndarray] = None,
                 monitor=None):
        self.train_data = train_data
        self.test_data = test_data
        self.monitor = monitor
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.train_data[Behavior.RANSOMWARE_POC][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]

    def sample_random_attack_state(self):
        """i.e. for starting state of an episode,
        (with replacement; it is possible that the same sample is chosen multiple times)"""
        rb = random.choice([b for b in Behavior if b != Behavior.NORMAL])
        attack_data = self.train_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=1), :]

    def sample_behaviour(self, b: Behavior):
        behavior_data = self.train_data[b]
        return behavior_data[np.random.randint(behavior_data.shape[0], size=1), :]

    def step(self, action: int):

        current_behaviour = self.current_state.squeeze()[-1]
        if self.monitor is None:
            if current_behaviour in supervisor_map[action]:
                # print("correct mtd chosen according to supervisor")
                new_state = self.sample_behaviour(Behavior.NORMAL)
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                # print("incorrect mtd chosen according to supervisor")
                new_state = self.sample_behaviour(current_behaviour)
                reward = self.calculate_reward(False)
                isTerminalState = False

        else:
            # would integrate a monitoring component here for a live system
            # new_state = self.monitor.get_current_behavior(),
            # but reward would need to be calculated by autoencoder
            reward = None

        return new_state, reward, isTerminalState

    def reset(self):
        self.current_state = self.sample_random_attack_state()
        self.reward = 0
        self.done = False
        return self.current_state

    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    def calculate_reward(self, success):
        """
        if current_behavior == supervisor_map[action]:
        then return positive
        else return negative

        this method can be exchanged for the online/unsupervised RL system with the autoencoder
        """
        if success:
            return 1
        else:
            return -1
