from typing import Dict, Tuple, List
from collections import defaultdict
from custom_types import Behavior, MTDTechnique
from autoencoder import AutoEncoderInterpreter
from scipy import stats
from tabulate import tabulate
import torch
import numpy as np
import pandas as pd
import os
import random

# define MTD - (target Attack) Mapping
# indices corresponding to sequence
actions = (MTDTechnique.CNC_IP_SHUFFLE, MTDTechnique.ROOTKIT_SANITIZER,
           MTDTechnique.RANSOMWARE_DIRTRAP, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE)

supervisor_map: Dict[int, Tuple[Behavior]] = defaultdict(lambda: -1, {
    # MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    0: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK, Behavior.NORMAL),
    1: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK, Behavior.NORMAL),
    2: (Behavior.RANSOMWARE_POC, Behavior.NORMAL),
    3: (Behavior.RANSOMWARE_POC, Behavior.NORMAL)
})


# TODO remove test_data, factor out environment core func
# handles the unsupervised, online-simulation of episodes
class SensorEnvironment:

    def __init__(self, train_data: Dict[Behavior, np.ndarray] = None,
                 interpreter: AutoEncoderInterpreter = None, state_samples=1):
        self.state_samples_ae = state_samples
        self.train_data = train_data
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.train_data[Behavior.RANSOMWARE_POC][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]
        self.interpreter = interpreter
        self.reset_to_behavior = None

    def sample_random_attack_state(self):
        """i.e. for starting state of an episode,
        (with replacement; it is possible that the same sample is chosen multiple times)"""
        rb = random.choice(
            [b for b in Behavior if b != Behavior.NORMAL and b != Behavior.ROOTKIT_BEURK])
        attack_data = self.train_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=1), :]

    def sample_behavior(self, b: Behavior):
        behavior_data = self.train_data[b]
        return behavior_data[np.random.randint(behavior_data.shape[0], size=1), :]

    def step(self, action: int):

        current_behavior = self.current_state.squeeze()[-1]

        if current_behavior in supervisor_map[action]:
            # print("correct mtd chosen according to supervisor")
            new_state = self.sample_behavior(Behavior.NORMAL)
            # ae predicts too many false positives: episode should not end, but behavior is normal (because MTD was correct)
            # note that this should not happen, as ae should learn to recognize normal behavior with near perfect accuracy
            if self.state_samples_ae > 1:
                for i in range(self.state_samples_ae - 1):  # real world simulation with multiple samples monitored
                    new_state = np.vstack((new_state, self.sample_behavior(Behavior.NORMAL)))
            # False Positive
            if torch.sum(self.interpreter.predict(new_state[:, :-1].astype(np.float32))) / len(new_state) > 0.5:
                # raise UserWarning("Should not happen! AE fails to predict majority of normal samples")
                reward = self.calculate_reward(False)
                isTerminalState = False
            # True Negative
            else:
                reward = self.calculate_reward(True)
                isTerminalState = True
            if self.state_samples_ae > 1:
                new_state = np.expand_dims(new_state[0, :], axis=0)  # throw away all but one transition for better decorrelation
        else:
            # print("incorrect mtd chosen according to supervisor")
            new_state = self.sample_behavior(current_behavior)
            if self.state_samples_ae > 1:
                for i in range(self.state_samples_ae - 1):  # real world simulation with multiple samples monitored
                    new_state = np.vstack((new_state, self.sample_behavior(current_behavior)))
            # False Negative
            # ae predicts a false negative: episode should end,  but behavior is not normal (because MTD was incorrect)
            # in this case, the next episode should start again with current_behavior
            if torch.sum(self.interpreter.predict(new_state[:, :-1].astype(np.float32))) / len(new_state) < 0.5:
                self.reset_to_behavior = current_behavior
                reward = self.calculate_reward(True)
                isTerminalState = True
            # True Positive
            else:
                reward = self.calculate_reward(False)
                isTerminalState = False
            if self.state_samples_ae > 1:
                new_state = np.expand_dims(new_state[0, :], axis=0)  # throw away all but one transition for better decorrelation

        self.current_state = new_state

        return new_state, reward, isTerminalState

    def reset(self):
        # in case of wrongful termination of an episode due to a false negative,
        # next episode should start with the given behavior again
        if self.reset_to_behavior:
            print(f"Resetting to behavior: {self.reset_to_behavior}")
            self.current_state = self.sample_behavior(self.reset_to_behavior)
            self.reset_to_behavior = None
        else:
            self.current_state = self.sample_random_attack_state()

        return self.current_state

    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    def calculate_reward(self, success):
        """this method can be refined to distinguish particularly wasteful/beneficial mtds"""
        if success:
            return 1
        else:
            return -1
