from typing import Dict, Tuple
from collections import defaultdict
from custom_types import Behavior, MTDTechnique
from scipy import stats
from tabulate import tabulate
import numpy as np
import pandas as pd
import os
from data_manager import DataManager



# define MTD - Attack Mapping
# TODO: Multiple attacks to same MTD, same attack to multiple MTD, i.e. Ransomware?
supervisor_map: Dict[MTDTechnique, Tuple[Behavior]] = defaultdict(lambda: (Behavior.NORMAL,), {
    MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    MTDTechnique.CNC_IP_SHUFFLE: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK),
    MTDTechnique.ROOTKIT_SANITIZER: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK),
    MTDTechnique.RANSOMWARE_DIRTRAP: (Behavior.RANSOMWARE_POC,),
    MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: (Behavior.RANSOMWARE_POC,)
})



# handles the supervised, online-simulation of episodes
class SensorEnvironment:

    def __init__(self, all_data: pd.DataFrame, monitor=None):
        self.data = all_data
        self.monitor = monitor
        self.current_state: pd.DataFrame = None


    def sample_random_state(self):
        """i.e. for starting state of an episode"""
        return self.data.sample()

    def sample_behaviour(self, b: Behavior):
        sample = self.sample_random_state()
        while sample.iloc[0]["attack"] != b.value:
            sample = self.sample_random_state()
        return sample


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
            if current_behaviour in [b.value for b in supervisor_map[action]]:
                print("correct mtd chosen according to supervisor")
                # TODO: return normal sample as new state
                new_state = self.sample_behaviour(Behavior.NORMAL)
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                new_state = self.sample_behaviour(current_behaviour)
                reward = self.calculate_reward(False)
                isTerminalState = False

        else:
            # would integrate a monitoring component here for a live system
            # new_state = self.monitor.get_current_behavior()
            reward = None

        return new_state, reward, isTerminalState


    def reset(self):
        self.current_state = self.sample_random_state()
        self.reward = 0
        self.done = False

        pass


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
