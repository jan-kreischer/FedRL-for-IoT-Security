from typing import Dict, Tuple, List
from src.custom_types import Behavior, MTDTechnique, actions, mitigated_by
import numpy as np
import random
from functools import reduce


# handles the supervised, online-simulation of episodes
class Environment:

    def __init__(self, entity_id, train_data: Dict[Behavior, np.ndarray] = None, sampling_probabilities: Dict[Behavior, int] = None, verbose=False):
        #print("Recognized Behaviours")
        #print(train_data.keys())
        self.entity_id = entity_id
        self.train_data = train_data
        
        if sampling_probabilities != None:
            sum_of_percentages = reduce(lambda x, y: x+y, sampling_probabilities.values())
            assert round(sum_of_percentages, 4) == 1, f"Make sure that all percentages sum to 100. Right now it is {sum_of_percentages}"
        self.sampling_probabilities = sampling_probabilities
        
        self.current_state: np.array = None
        self.observation_space_size: int = len(self.train_data[Behavior.NORMAL][0][:-1])
        self.actions: List[int] = [i for i in range(len(actions))]
        
        self.verbose=False

    # Returns a randomly selected attack state with non normal behaviour.
    def sample_random_attack_state(self):
        if self.sampling_probabilities != None:
            behaviors = list(self.sampling_probabilities.keys())
            attacks = [b for b in behaviors if b != Behavior.NORMAL]
            attacks = behaviors
            sampling_probabilities = self.sampling_probabilities.values()
            sampled_attack = random.choices(attacks, weights=sampling_probabilities, k=1)[0]
            attack_states = self.train_data[sampled_attack]
            return attack_states[np.random.randint(attack_states.shape[0], size=1), :]
        else:
            sampled_attack = random.choice([b for b in self.train_data.keys() if b != Behavior.NORMAL])

        attack_states = self.train_data[sampled_attack]
        return attack_states[np.random.randint(attack_states.shape[0], size=1), :]
    
    # Return random sample with specified behaviour
    def sample_behavior(self, b: Behavior):
        behavior_data = self.train_data[b]
        return behavior_data[np.random.randint(behavior_data.shape[0], size=1), :]


    def step(self, action: int):
        current_behavior = self.current_state.squeeze()[-1]

        if current_behavior in mitigated_by[action]:
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