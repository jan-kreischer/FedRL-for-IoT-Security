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

normal_afterstates = [(Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE),
                      (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP),
                      (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE),
                      (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER)]


# handles the supervised, online-simulation of episodes using decision and afterstate data
class SensorEnvironment:
    # training data split/test data
    def __init__(self, decision_train_data: Dict[Behavior, np.ndarray] = None,
                 decision_test_data: Dict[Behavior, np.ndarray] = None,
                 after_train_data: Dict[Tuple[Behavior, MTDTechnique], np.ndarray] = None,
                 after_test_data: Dict[Tuple[Behavior, MTDTechnique], np.ndarray] = None,
                 interpreter: AutoEncoderInterpreter = None, state_samples = 1):
        self.dtrain_data = decision_train_data
        self.dtest_data = decision_test_data
        self.atrain_data = after_train_data
        self.atest_data = after_test_data
        self.state_samples_ae = state_samples
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.dtrain_data[Behavior.RANSOMWARE_POC][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]

        self.interpreter = interpreter
        self.reset_to_behavior = None

    def sample_initial_decision_attack_state(self):
        # in case of wrongful termination of an episode due to a false negative,
        # next episode should start with the decision state of the given behavior again
        if self.reset_to_behavior:
            print(f"Resetting to behavior: {self.reset_to_behavior}")
            attack_data = self.dtrain_data[self.reset_to_behavior]
            self.reset_to_behavior = None
        else:
            rb = random.choice([b for b in Behavior if b != Behavior.NORMAL and b != Behavior.ROOTKIT_BEURK and b != Behavior.CNC_THETICK])
            attack_data = self.dtrain_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=1), :]

    def sample_afterstate(self, b: Behavior, m: MTDTechnique):
        after_data = self.atrain_data[(b, m)]
        return after_data[np.random.randint(after_data.shape[0], size=1), :]

    def step(self, action: int):
        current_behavior = self.current_state.squeeze()[-1] if self.current_state.squeeze()[-1] in Behavior else \
        self.current_state.squeeze()[-2]
        #print(f"current behavior = {current_behavior}")
        chosen_mtd = actions[action]


        if current_behavior in supervisor_map[action] or (current_behavior, chosen_mtd) in normal_afterstates:
            #print("correct mtd chosen according to supervisor")
            new_state = self.sample_afterstate(current_behavior, chosen_mtd)

            # ae predicts too many false positives: episode should not end, but behavior is normal (because MTD was correct)
            # note that this should not happen, as ae should learn to recognize normal behavior with near perfect accuracy
            if self.interpreter:
                if self.state_samples_ae > 1:
                    for i in range(self.state_samples_ae - 1):  # real world simulation with multiple samples monitored
                        new_state = np.vstack((new_state, self.sample_afterstate(current_behavior, chosen_mtd)))
                if torch.sum(self.interpreter.predict(new_state[:, :-2].astype(np.float32))) / len(new_state) > 0.5:
                    # raise UserWarning("Should not happen! AE fails to predict majority of normal samples! Too many False Positives!")
                    reward = self.calculate_reward(False)
                    isTerminalState = False
                else:
                    reward = self.calculate_reward(True)
                    isTerminalState = True
                if self.state_samples_ae > 1:
                    new_state = np.expand_dims(new_state[0, :], axis=0)  # throw away all but one transition for better decorrelation
        else:
            # print("incorrect mtd chosen according to supervisor")
            new_state = self.sample_afterstate(current_behavior, chosen_mtd)
            # ae predicts a false negative: episode should end,  but behavior is not normal (because MTD was incorrect)
            # in this case, the next episode should start again with current_behavior
            if self.interpreter and self.interpreter.predict(new_state[:, :-2].astype(np.float32)) == 0:
                self.reset_to_behavior = current_behavior
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                reward = self.calculate_reward(False)
                isTerminalState = False

        self.current_state = new_state
        return new_state, reward, isTerminalState

    def reset(self):
        self.current_state = self.sample_initial_decision_attack_state()
        return self.current_state

    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    def calculate_reward(self, success):
        """this method can be exchanged for the online/unsupervised RL system with the autoencoder"""
        if success:
            return 1
        else:
            return -1
