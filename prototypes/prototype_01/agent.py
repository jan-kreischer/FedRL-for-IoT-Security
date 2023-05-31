'''
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from deep_q_network import DeepQNetwork
import copy

class Agent:
    def __init__(self, agent_id: int, input_dims: int, n_actions, batch_size,
                 lr, gamma, epsilon, eps_end=0.02, eps_dec=1e-4, buffer_size=100000, is_global_agent=False, osiose=False):
        self.agent_id = agent_id
        self.is_global_agent = is_global_agent
        #self.verbose = verbose
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_max = epsilon # Initial epsilon value
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.total_accuracies = {}
        self.mean_class_accuracies = {}

        self.episode_action_memory = set()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0.0], maxlen=100)  # for printing progress

        self.batch_size = batch_size

        self.online_net = DeepQNetwork(lr, n_actions=n_actions,
                                       input_dims=input_dims,
                                       fc1_dims=60, fc2_dims=30)
        self.target_net = DeepQNetwork(lr, n_actions=n_actions,
                                       input_dims=input_dims,
                                       fc1_dims=60, fc2_dims=30)
        self.target_net.load_state_dict(self.online_net.state_dict())

    def choose_action(self, observation):
        try:
            if np.random.random() > self.epsilon:
                #
                action = self.take_greedy_action(observation)
                if action in self.episode_action_memory:
                    action = np.random.choice(list(set(self.action_space).difference(self.episode_action_memory)))
            else:
                action = np.random.choice(list(set(self.action_space).difference(self.episode_action_memory)))
            self.episode_action_memory.add(action)
        except ValueError:
            return -1
        return action

    def take_greedy_action(self, observation):
        state = torch.from_numpy(observation.astype(np.float32)).to(self.online_net.device)
        actions = self.online_net.forward(state)
        action = torch.argmax(actions).item()
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

        # gradient descent
        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()

        #new_epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        # epsilon decay
        #print(f"{self.get_name()}: {round(self.epsilon, 4)}->{round(new_epsilon, 4)}")
        #self.epsilon = new_epsilon
   
    def epsilon_decay(self, nr_trained_episodes):
        #print(f"Episode {nr_trained_episodes} => {self.epsilon}")
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_weights(self):
        return copy.deepcopy(self.target_net.state_dict())
    
    def update_weights(self, model_params):
        #start_time = time_ns()
        self.online_net.load_state_dict(copy.deepcopy(model_params))
        self.target_net.load_state_dict(copy.deepcopy(model_params))
        #end_time = time_ns()
        #time_difference = end_time - start_time
        #print(f"Updating weights on {self.get_name()} took {time_difference / 10**9}s")
        
    def get_name(self):
        if self.agent_id == 0:
            return "Global Agent"
        else:
            return f"Agent {self.agent_id}"
    
    def save_agent_state(self, n: int, directory: str):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'batch_size': self.batch_size,
            'replay_buffer': self.replay_buffer,
            'reward_buffer': self.reward_buffer,
            'action_space': self.action_space,
            'gamma': self.gamma,
            'eps': self.epsilon,
            'eps_min': self.eps_min,
            'eps_dec': self.eps_dec,
            'lr': self.lr
        }, f"{directory}/trained_models/agent_{n}.pth")

        #torch.save(self.online_net.state_dict(), f"offline_prototype_2_raw_behaviors/trained_models/online_net_{n}.pth")
        #torch.save(self.target_net.state_dict(), f"offline_prototype_2_raw_behaviors/trained_models/target_net_{n}.pth")
'''       
    

from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from deep_q_network import DeepQNetwork
from src.custom_types import Behavior
#from deep_q_network import DeepQNetwork
import copy
#from src.custom_types import Behavior, MTDTechnique

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

        #if agent_id == 1:
        #    print(self.online_net)
        
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
        #print(action)
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
        return copy.deepcopy(self.target_net.state_dict())
    
    def update_weights(self, model_params):
        #start_time = time_ns()
        self.online_net.load_state_dict(copy.deepcopy(model_params))
        self.target_net.load_state_dict(copy.deepcopy(model_params))
        #end_time = time_ns()
        #time_difference = end_time - start_time
        #print(f"Updating weights on {self.get_name()} took {time_difference / 10**9}s")
        
    def get_name(self):
        if self.agent_id == 0:
            return "Global Agent"
        else:
            return f"Agent {self.agent_id}"
        
'''
        from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
#from deep_q_network import DeepQNetwork
import copy
#from src.custom_types import Behavior, MTDTechnique

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

        if agent_id == 1:
            print(self.online_net)
        
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
        #print(action)
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
        return copy.deepcopy(self.target_net.state_dict())
    
    def update_weights(self, model_params):
        #start_time = time_ns()
        self.online_net.load_state_dict(copy.deepcopy(model_params))
        self.target_net.load_state_dict(copy.deepcopy(model_params))
        #end_time = time_ns()
        #time_difference = end_time - start_time
        #print(f"Updating weights on {self.get_name()} took {time_difference / 10**9}s")
        
    def get_name(self):
        if self.agent_id == 0:
            return "Global Agent"
        else:
            return f"Agent {self.agent_id}"
'''