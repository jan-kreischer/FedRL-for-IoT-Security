from typing import Dict, Tuple
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
supervisor_map: Dict[MTDTechnique, Tuple[Behavior]] = defaultdict(lambda: (Behavior.NORMAL,), {
    MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    MTDTechnique.CNC_IP_SHUFFLE: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK),
    MTDTechnique.ROOTKIT_SANITIZER: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK),
    MTDTechnique.RANSOMWARE_DIRTRAP: (Behavior.RANSOMWARE_POC,),
    MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: (Behavior.RANSOMWARE_POC,)
})
actions = (MTDTechnique.CNC_IP_SHUFFLE, MTDTechnique.ROOTKIT_SANITIZER,
           MTDTechnique.RANSOMWARE_DIRTRAP, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE)



# handles the supervised, online-simulation of episodes
class SensorEnvironment:

    def __init__(self, all_data: Dict[Behavior, pd.DataFrame] = None, monitor=None):
        self.data = all_data
        self.monitor = monitor
        self.current_state: pd.DataFrame = None
        self.observation_space_size: int = len(self.data[Behavior.RANSOMWARE_POC].iloc[0])
        self.actions: int = [i for i in range(len(actions))]

    def sample_random_attack_state(self):
        """i.e. for starting state of an episode"""
        rb = random.choice([b for b in Behavior if b != Behavior.NORMAL])
        return self.data[rb].sample()

    def sample_behaviour(self, b: Behavior):
        return self.data[b].sample()

    def step(self, action: MTDTechnique):
        """
        reward: call calculate_reward(new_state)
        """
        new_state = None
        isTerminalState = False

        print(action.name)
        print(action.value)
        current_behaviour = self.current_state.iloc[0]["attack"]
        if self.monitor is None:
            if current_behaviour in supervisor_map[action]:
                print("correct mtd chosen according to supervisor")
                new_state = self.sample_behaviour(Behavior.NORMAL)
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                print("incorrect mtd chosen according to supervisor")
                new_state = self.sample_behaviour(current_behaviour)
                reward = self.calculate_reward(False)
                isTerminalState = False

        else:
            # would integrate a monitoring component here for a live system
            # new_state = self.monitor.get_current_behavior()
            reward = None

        return new_state, reward, isTerminalState

    def reset(self):
        self.current_state = self.sample_random_attack_state()
        self.reward = 0
        self.done = False




    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    def calculate_reward(self, success):
        """
        if action == supervisor_map[state.behavior]:
        then return positive
        else return negative

        this method can be exchanged for the online/unsupervised RL system with the autoencoder
        """
        if success:
            return 100
        else:
            return -100
