from autoencoder import AutoEncoder
import numpy as np
import torch.nn.functional as F

from torch import nn

class TiedAutoEncoder(AutoEncoder):
    

    def __init__(self, X_valid, X_test, y_test, evaluation_data, num_stds=[1], activation_function=torch.nn.ReLU(), batch_size: int = 64, verbose=False):
        
        super(TiedAutoEncoder, self).__init__(None, X_valid, X_test, y_test, evaluation_data)

        n_features = X_valid.shape[1]
        self.weight_matrix_1 = nn.Parameter(torch.randn(n_features, 30))
        self.weight_matrix_2 = nn.Parameter(torch.randn(30, 20))
        self.weight_matrix_3 = nn.Parameter(torch.randn(20, 10))
        
        
    def forward(self, x):
        x = F.relu(F.linear(x, self.weight_matrix_1.T))
        x = F.relu(F.linear(x, self.weight_matrix_2.T))
        x = F.relu(F.linear(x, self.weight_matrix_3.T))
        x = F.relu(F.linear(x, self.weight_matrix_3))
        x = F.relu(F.linear(x, self.weight_matrix_2))
        x = F.relu(F.linear(x, self.weight_matrix_1))
        return x