import torch 
from torch import nn

def initial_autoencoder_architecture(n_features):
    #return nn.Sequential(
    #    nn.Linear(n_features, 23),
    #    nn.BatchNorm1d(23),
    #    nn.GELU(),
    #    nn.Linear(23, 11),
    #    nn.GELU(),
    #    nn.Linear(11, 23),
    #    nn.BatchNorm1d(23),
    #    nn.GELU(),
    #    nn.Linear(23, n_features,),
    #    nn.GELU()
    #)
    
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Linear(64, 16),
        nn.GELU(),
        #nn.Linear(32, 16),
        #nn.GELU(),
        #nn.Linear(16, 32),
        #nn.GELU(),
        nn.Linear(16, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Linear(64, n_features),
        nn.GELU()
    )

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

