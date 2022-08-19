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
    0: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK),
    1: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK),
    2: (Behavior.RANSOMWARE_POC,),
    3: (Behavior.RANSOMWARE_POC,)
})


# handles the supervised, online-simulation of episodes using decision and afterstate data

# TODO: train_data and test_data must contain
#  1. decision states for all attacks
#  2. afterstates for all behavior/mtd combinations
class SensorEnvironment:
    # training data split/test data
    def __init__(self, decision_train_data: Dict[Behavior, np.ndarray] = None,
                 decision_test_data: Dict[Behavior, np.ndarray] = None,
                 after_train_data: Dict[Tuple[Behavior, MTDTechnique], np.ndarray] = None,
                 after_test_data: Dict[Tuple[Behavior, MTDTechnique], np.ndarray] = None,
                 monitor=None, interpreter: AutoEncoderInterpreter = None):
        self.dtrain_data = decision_train_data
        self.dtest_data = decision_test_data
        self.atrain_data = after_train_data
        self.atest_data = after_test_data

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
            rb = random.choice([b for b in Behavior if b != Behavior.NORMAL])
            attack_data = self.dtrain_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=1), :]

    def sample_afterstate(self, b: Behavior, m: MTDTechnique):
        after_data = self.atrain_data[(b,m)]
        return after_data[np.random.randint(after_data.shape[0], size=1), :]

    def step(self, action: int):
        current_behavior = self.current_state.squeeze()[-1]
        print(f"current behavior = {current_behavior}")
        chosen_mtd = actions[action]

        if current_behavior in supervisor_map[action]:
            # print("correct mtd chosen according to supervisor")
            new_state = self.sample_afterstate(Behavior.NORMAL, chosen_mtd)

            # ae predicts too many false positives: episode should not end, but behavior is normal (because MTD was correct)
            # note that this should not happen, as ae should learn to recognize normal behavior with near perfect accuracy
            if self.interpreter:
                for i in range(9):  # real world simulation with 10 samples
                    new_state = np.vstack((new_state, self.sample_afterstate(Behavior.NORMAL, chosen_mtd)))
                if torch.sum(self.interpreter.predict(new_state[:, :-1].astype(np.float32))) / len(new_state) > 0.5:
                    raise UserWarning("Should not happen! AE fails to predict majority of normal samples")
                    # reward = self.calculate_reward(False)
                    # isTerminalState = False
                else:
                    new_state = np.expand_dims(new_state[0, :], axis=0)  # throw away all but one transition
                    reward = self.calculate_reward(True)
                    isTerminalState = True
        else:
            # print("incorrect mtd chosen according to supervisor")
            new_state = self.sample_afterstate(current_behavior, chosen_mtd)
            # ae predicts a false negative: episode should end,  but behavior is not normal (because MTD was incorrect)
            # in this case, the next episode should start again with current_behavior
            if self.interpreter and self.interpreter.predict(new_state[:, :-1].astype(np.float32)) == 0:
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
