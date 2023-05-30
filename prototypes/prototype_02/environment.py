# Sampling from distribution has to be added here.       
from typing import Dict, Tuple, List
from collections import defaultdict
from src.custom_types import Behavior, MTDTechnique, Execution, Evaluation, actions, mitigated_by
from scipy import stats
from tabulate import tabulate
import torch
import numpy as np
import pandas as pd
import os
import random
from functools import reduce

from src.autoencoder import AutoEncoder

class Environment:

    def __init__(self, 
                 environment_id: int,
                 training_data: Dict[Behavior, np.ndarray] = None,
                 state_interpreter: AutoEncoder = None,
                 n_state_samples: int = 1,
                 normality_probability: float = 0.5,
                 sampling_probabilities: Dict[Behavior, float] = None,
                 verbose=False):
        
        #print(f"len(actions): {len(actions)}")
        self.environment_id = environment_id
        self.n_state_samples = n_state_samples
        self.training_data = training_data
        self.normality_probability = normality_probability
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.training_data[list(self.training_data.keys())[0]][0][:-1])
        self.actions: List[int] = [i for i in range(len(MTDTechnique))]
        self.state_interpreter = state_interpreter
        self.reset_to_behavior = None
        
        self.nr_fps = 0
        self.nr_fns = 0
        self.total_nr_steps = 0
        self.total_nr_episodes = 0
        
        #print(normality_probability)
        #for key, value in training_data.items():
        #    print(f"{key}: {len(value)}")
        if sampling_probabilities != None:
            sum_of_percentages = reduce(lambda x, y: x+y, sampling_probabilities.values())
            assert round(sum_of_percentages, 4) == 1, f"Make sure that all percentages sum to 100. Right now it is {sum_of_percentages}"
        self.sampling_probabilities = sampling_probabilities
        
        self.verbose = verbose

    def sample_initial_decision_state(self):
        """i.e. for starting state of an episode,
        (with replacement; it is possible that the same sample is chosen multiple times)"""
        #if np.random.random_sample() < self.normality_probability:
        #    #print("sampling normal behavior")
        #    sampled_behavior = Behavior.NORMAL
        #    #print("Using normal behavior")
        #else:
        if self.sampling_probabilities != None: 
            # Sample according to given sampling probabilities
            attacks = [b for b in list(self.sampling_probabilities.keys())]
            sampling_probabilities = self.sampling_probabilities.values()
            sampled_behavior = random.choices(attacks, weights=sampling_probabilities, k=1)[0]
        else:
            # Randomly sample an attack
            sampled_behavior = random.choice([behavior for behavior in self.training_data.keys()])

        # All samples for the chosen behavior
        state_samples = self.training_data[sampled_behavior]
        
        # Select only one sample to return for the chosen behavior
        selected_state_sample = state_samples[np.random.randint(state_samples.shape[0], size=self.n_state_samples), :]
        return selected_state_sample

    def sample_behavior(self, b: Behavior):
        behavior_data = self.training_data[b]
        return behavior_data[np.random.randint(behavior_data.shape[0], size=self.n_state_samples), :]

    def step(self, action: int):
        self.total_nr_steps+=1
        
        #self.reset_to_behavior = None
        
        current_behavior = self.current_state[0, -1]
        #print(current_behavior)
        #print(f"action: {action}")
        #print(f"current_behavior: {current_behavior}")
        #print(f"mitigated_by[action]: {mitigated_by[action]}")
        if current_behavior in mitigated_by[action]:
            # Correct MTD was chosen
            # => Afterstate Normal
            #if self.verbose:
            #    print("Correct MTD chosen according to supervisor")
            next_state = (self.sample_behavior(Behavior.NORMAL))
            
            # ae predicts too many false positives: episode should not end, but behavior is normal (because MTD was correct)
            # note that this should not happen, as ae should learn to recognize normal behavior with near perfect accuracy

            if self.state_is_classified_normal(next_state[:, :-1].astype(np.float32)):
                # True Negative (TN)
                # Afterstate Normal, Classified Normal
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                # raise UserWarning("Should not happen! AE fails to predict majority of normal samples")
                # False Positive (FP)
                # Afterstate Normal, Classified Abnormal
                self.nr_fps+=1
                if self.verbose:
                    print(f"False Positive: ({current_behavior} misclassified as ABNORMAL) by anomaly detector")
                reward = self.calculate_reward(False)
                #reward=1
                isTerminalState = False

        else:
            # Incorrect MTD was chosen
            # => Afterstate Abnormal
            #if self.verbose:
            #    print("Incorrect MTD chosen according to supervisor")
            
            next_state = (self.sample_behavior(current_behavior))
            
            # False Negative
            # ae predicts a false negative: episode should end,  but behavior is not normal (because MTD was incorrect)
            # in this case, the next episode should start again with current_behavior

            if self.state_is_classified_normal(next_state[:, :-1].astype(np.float32)):
                # False Negative (FN)
                # Afterstate Abnormal, Classified Normal
                if self.verbose and next_state != Behavior.NORMAL:
                    self.nr_fns+=1
                    print(f"False Negative: ({current_behavior} misclassified as NORMAL) by anomaly detector")
                    self.reset_to_behavior = current_behavior
                reward = self.calculate_reward(True)
                isTerminalState = True
            else:
                # True Positive (TP)
                # Afterstate Abnormal, Classified Abnormal
                #if self.verbose & next_state == Behavior.NORMAL:
                #    print(f"False Negative: ({current_behavior} misclassified as NORMAL) by anomaly detector")
                reward = self.calculate_reward(False)
                # reward=-1
                isTerminalState = False

        self.current_state = next_state
        if self.n_state_samples > 1:
            next_state = np.expand_dims(next_state[0, :], axis=0)  # throw away all but one transition for better decorrelation

        return next_state, reward, isTerminalState

    def state_is_classified_normal(self, state_samples):
        # Classified as normal for < 0.5
        # Otherwise considered abnormal
        return (torch.sum(self.state_interpreter.predict(state_samples, n_std=20)) / len(state_samples)) <= 0.5
      
    '''
        
    '''
    def reset(self):
        #print(f"resetting to {self.reset_to_behavior}")
        # In case of wrongful termination of an episode due to a false negative,
        # next episode should start with the given behavior again
        if self.reset_to_behavior:
            #if self.verbose:
            #print(f"Called in {self.total_nr_steps}, Resetting to behavior: {self.reset_to_behavior}")
            self.current_state = self.sample_behavior(self.reset_to_behavior)
            self.reset_to_behavior = None
            # WARNING:
            # if the behavior to reset to is never detected as an anomaly,
            # it could get stuck in an endless loop here
        else:
            self.current_state = self.sample_initial_decision_state()

        #b = self.current_state[0, -1]
        #if not self.state_is_classified_normal(self.current_state[:, :-1].astype(np.float32)):
        #    # FP/TP - start training
        #    # below must be here, otherwise it's possible that there is a false negative and the next episode starts with a different behavior
        #    self.reset_to_behavior = None
        #    break
        
        #print(f"did reset to {self.current_state[:,-1]}")
        return np.expand_dims(self.current_state[0, :], axis=0)

    def calculate_reward(self, success):
        if success:
            return 1
        else:
            return -1
        
    def plot_fn_ratio(self):
        print(f"{self.nr_fns} FNs / {self.total_nr_steps} Steps ... {round(self.nr_fns/self.total_nr_steps, 4)}")  
        
    def plot_fp_ratio(self):
        print(f"{self.nr_fps} FPs /{self.total_nr_steps} Steps ... {round(self.nr_fps/self.total_nr_steps, 4)}")    