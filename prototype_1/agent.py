from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# TODO:
#  - add softmax layer at the end of dqn
#  - check code with debugger, run through, understanding of each loc
#  - analyze reward buffer for first 100/200 contents
#  - min max scaling on input?
#  - how do buffer size and batch size belong together
#  - try removing the first few non-perf features
#  - how can last step before end be valued most?
#  - try adjusting the reward -> 0 for fail, 1 for success
#  - reduce/increase exploration/epsilon
#  - adapt hidden layers/size

# TODO: Optimization
#  - integrate that an agent may not repeatedly select MTDs that have not worked!
#  -> add a buffer of already utilized techniques in an episode!
#  -> adapt choose_action!
#  - ensure that most resource-consuming MTDs are penalized harder than such that are not
#  (i.e. -1 for dirtrap, -0.8 for ipshuffle, -0.7 filext, -0.5rootkit), because dirtrap is computationally intense, ipshuffle results in downtime, rootkit lasts miliseconds




class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions



class Agent:
    def __init__(self, input_dims, n_actions, batch_size,
                 lr, gamma, epsilon, eps_end=0.02, eps_dec=1e-4, buffer_size=100000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]

        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0.0], maxlen=100) # for printing progress

        self.batch_size = batch_size

        self.online_net = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.target_net = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.target_net.load_state_dict(self.online_net.state_dict())


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.from_numpy(observation.astype(np.float32)).to(self.online_net.device)
            actions = self.online_net.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
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
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = t_rewards + self.gamma * (1 - t_dones) * max_target_q_values

        # compute loss
        q_values = self.online_net(t_obses)
        taken_action_q_values = torch.gather(input=q_values, dim=1, index=t_actions.unsqueeze(1))

        loss = self.online_net.loss(taken_action_q_values, targets).to(self.target_net.device)

        # gradient descent
        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()

        # epsilon decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def derive_reward_from_new_state(self, new_state):
        """method only needed in unsupervised setting"""
        # TODO: call autoencoder here and check for normality
        pass


    def save_dqns(self, n: int):
        torch.save(self.online_net.state_dict(), f"trained_models/online_net_{n}.pth")
        torch.save(self.target_net.state_dict(), f"trained_models/target_net_{n}.pth")
