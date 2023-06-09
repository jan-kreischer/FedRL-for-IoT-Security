import copy
import torch
import random
import numpy as np
from torch import nn
from typing import Dict
from collections import deque
import torch.nn.functional as F

from src.custom_types import Behavior, MTDTechnique

from src.deep_q_network import DeepQNetwork


class Agent:
    def __init__(self, 
                 agent_id: int, 
                 deep_q_network: DeepQNetwork,
                 batch_size,
                 gamma: float,
                 optimizer,
                 eps,
                 eps_min,
                 eps_dec,
                 buffer_size=100000
                ):
        self.agent_id = agent_id

        self.online_net = deep_q_network
        self.target_net = copy.deepcopy(deep_q_network)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.gamma = gamma
        
        self.optimizer = optimizer
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        
        self.action_space = [i for i in range(deep_q_network.n_actions)]
        
        self.total_accuracies = {}
        self.mean_class_accuracies = {}
        
        self.behavior_accuracies = {}
        for behavior in Behavior:
            self.behavior_accuracies[behavior] = {}

        self.episode_action_memory = set()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0.0], maxlen=100)  # for printing progress

        self.batch_size = batch_size

        self.test_accuracies = {}
        
        self.losses = []

    def choose_action(self, observation):
        try:
            if np.random.random() > self.eps:
                #
                action = self.take_greedy_action(observation)
                if action in self.episode_action_memory:
                    action = self.take_random_action()
            else:
                action = self.take_random_action()
            self.episode_action_memory.add(action)
        except ValueError:
            return -1
        
        return action

    def take_random_action(self):
        return np.random.choice(list(set(self.action_space).difference(self.episode_action_memory)))
        
    def take_greedy_action(self, state):
        state = torch.from_numpy(state.astype(np.float32)).to(self.online_net.device)
        q_values = self.online_net.forward(state)
        action = torch.argmax(q_values).item()
        return action

    def learn(self):
        # init data batch from memory replay for dqn
        transitions = random.sample(self.replay_buffer, self.batch_size)
        b_obses = np.stack([t[0].astype(np.float32).squeeze(0) for t in transitions], axis=0)
        b_actions = np.asarray([t[1] for t in transitions]).astype(np.int64)
        b_rewards = np.asarray([t[2] for t in transitions]).astype(np.int16)
        b_new_obses = np.stack([t[3].astype(np.float32).squeeze(0) for t in transitions], axis=0)
        b_dones = np.asarray([t[4] for t in transitions]).astype(np.int16)
        t_obses = torch.from_numpy(b_obses).to(self.target_net.device)
        t_actions = torch.from_numpy(b_actions).to(self.target_net.device)
        t_rewards = torch.from_numpy(b_rewards).to(self.target_net.device)
        t_new_obses = torch.as_tensor(b_new_obses).to(self.target_net.device)
        t_dones = torch.as_tensor(b_dones).to(self.target_net.device)

        # compute targets
        target_q_values = self.target_net(t_new_obses)
        max_target_q_values = torch.max(target_q_values, dim=1)[0]

        targets = (t_rewards + self.gamma * (1 - t_dones) * max_target_q_values).unsqueeze(1)

        # compute loss
        q_values = self.online_net(t_obses)
        taken_action_q_values = torch.gather(input=q_values, dim=1, index=t_actions.unsqueeze(1))

        loss = self.online_net.loss(taken_action_q_values, targets).to(self.target_net.device)
        self.losses.append(loss.item())
        
        # gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def epsilon_decay(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_weights(self):
        return copy.deepcopy(self.online_net.state_dict())
    
    def update_weights(self, model_params):
        self.online_net.load_state_dict(copy.deepcopy(model_params))
        self.target_net.load_state_dict(copy.deepcopy(model_params))
        
    def get_name(self):
        if self.agent_id == 0:
            return "Global Agent"
        else:
            return f"Agent {self.agent_id}"

