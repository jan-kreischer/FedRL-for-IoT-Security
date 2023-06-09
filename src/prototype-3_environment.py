from typing import Dict, Tuple, List
from collections import defaultdict
from src.custom_types import Behavior, MTDTechnique, actions, supervisor_map, normal_afterstates
from src.autoencoder import AutoEncoderInterpreter
from scipy import stats
from tabulate import tabulate
import torch
import numpy as np
import pandas as pd
import os
import random


# handles the supervised, online-simulation of episodes using decision and afterstate data
class SensorEnvironment:
    # training data split/test data
    def __init__(self, decision_train_data: Dict[Behavior, np.ndarray] = None,
                 after_train_data: Dict[Tuple[Behavior, MTDTechnique], np.ndarray] = None,
                 interpreter: AutoEncoderInterpreter = None, state_samples=1, normal_prob=0.8):
        self.dtrain_data = decision_train_data
        self.atrain_data = after_train_data
        self.num_state_samples = state_samples
        self.normal_prob = normal_prob
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.dtrain_data[Behavior.RANSOMWARE_POC][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]

        self.interpreter = interpreter
        self.reset_to_behavior = None

    def sample_initial_decision_state(self):
        if np.random.random_sample() < self.normal_prob:
            attack_data = self.dtrain_data[Behavior.NORMAL]
        else:
            rb = random.choice([b for b in Behavior if b != Behavior.NORMAL])
            attack_data = self.dtrain_data[rb]
        return attack_data[np.random.randint(attack_data.shape[0], size=self.num_state_samples), :]

    def sample_afterstate(self, b: Behavior, m: MTDTechnique):
        after_data = self.atrain_data[(b, m)]
        return after_data[np.random.randint(after_data.shape[0], size=self.num_state_samples), :]

    def step(self, action: int):
        if self.current_state[0, -1] in Behavior:
            current_behavior = self.current_state[0, -1]
            prev_mtd = None
        else:
            current_behavior = self.current_state[0, -2]
            prev_mtd = self.current_state[0, -1]

        # print(f"current behavior = {current_behavior}")
        chosen_mtd = actions[action]

        # Theoretically there is a bug here regarding process flow, see explanation
        if current_behavior in supervisor_map[action] or (current_behavior, prev_mtd) in normal_afterstates:
            # print("correct mtd chosen according to supervisor")

            # condition below ensures that if the previous mtd has already been successful, but the episode did not end
            # (due to a false positive), then the Behavior will be set to normal+the new mtd.
            # A normal+chosen_mtd afterstate models reality most accurately in this case.
            # For the case of not having yet chosen the correct mtd, afterstates are sampled according to
            # current behavior and chosen mtd
            if (current_behavior, prev_mtd) in normal_afterstates:
                new_state = self.sample_afterstate(Behavior.NORMAL, chosen_mtd)
            else:
                new_state = self.sample_afterstate(current_behavior, chosen_mtd)

            # ae predicts too many false positives: episode should not end, but behavior is normal (because MTD was correct)
            # note that this should not happen, as ae should learn to recognize normal behavior with near perfect accuracy
            if torch.sum(self.interpreter.predict(new_state[:, :-2].astype(np.float32))) / len(new_state) > 0.5:
                # raise UserWarning("Should not happen! AE fails to predict majority of normal samples! Too many False Positives!")
                reward = self.calculate_reward(False)
                isTerminalState = False
            else:
                reward = self.calculate_reward(True)
                isTerminalState = True
        else:
            # print("incorrect mtd chosen according to supervisor")
            new_state = self.sample_afterstate(current_behavior, chosen_mtd)
            # ae predicts a false negative: episode should end,  but behavior is not normal (because MTD was incorrect)
            # in this case, the next episode should start again with current_behavior
            if torch.sum(self.interpreter.predict(new_state[:, :-2].astype(np.float32))) / len(new_state) < 0.5:
                self.reset_to_behavior = current_behavior
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                reward = self.calculate_reward(False)
                isTerminalState = False

        self.current_state = new_state
        if self.num_state_samples > 1:
            new_state = np.expand_dims(new_state[0, :],
                                       axis=0)  # throw away all but one transition for better decorrelation

        return new_state, reward, isTerminalState

    def reset(self):
        while True:
            # in case of wrongful termination of an episode due to a false negative,
            # next episode should start with the given behavior again
            if self.reset_to_behavior:
                print(f"Resetting to behavior: {self.reset_to_behavior}")
                attack_data = self.dtrain_data[self.reset_to_behavior]
                self.current_state = attack_data[np.random.randint(attack_data.shape[0], size=self.num_state_samples),
                                     :]
                # WARNING:
                # if the behavior to reset to is never detected as an anomaly,
                # it could get stuck in an endless loop here
            else:
                self.current_state = self.sample_initial_decision_state()

            b = self.current_state[0, -1]

            if (torch.sum(self.interpreter.predict(self.current_state[:, :-1].astype(np.float32))) / len(
                    self.current_state) > 0.5):
                # FP/TP - start training
                # below must be here, otherwise it's possible that there is a false negative and the next episode starts with a different behavior
                self.reset_to_behavior = None
                break

        return np.expand_dims(self.current_state[0, :], axis=0)

    # TODO: possibly adapt to distinguish between MTDs that are particularly wasteful in case of wrong deployment
    #  Ensure that most resource-consuming MTDs are penalized harder than such that are not
    #  (i.e. -1 for dirtrap, -0.8 for ipshuffle, -0.7 filext, -0.5rootkit),
    #  because:
    #  - dirtrap is computationally intense,
    #  - ipshuffle results in downtime,
    #  - rootkit lasts miliseconds

    def calculate_reward(self, success):
        """this method can be exchanged for the online/unsupervised RL system with the autoencoder"""
        if success:
            return 1
        else:
            return -1
