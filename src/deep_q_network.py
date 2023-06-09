from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import dropout


class DeepQNetwork(nn.Module):
    def __init__(self, 
                 n_features: int,
                 n_hidden_1: int,
                 n_hidden_2: int,
                 n_hidden_3: int,
                 n_actions: int,
                 loss=nn.MSELoss(),
                ):
        super(DeepQNetwork, self).__init__()
        
        # Input-Output
        self.n_features = n_features
        self.n_actions = n_actions

        # Layers
        self.L1 = nn.Linear(n_features, n_hidden_1)
        self.L2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.L3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.L4 = nn.Linear(n_hidden_2, n_actions)
        
        self.loss = loss
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = F.selu(self.L1(x))
        x = F.selu(self.L2(x))
        #x = F.selu(self.L3(x))
        x = self.L4(x)
        return x