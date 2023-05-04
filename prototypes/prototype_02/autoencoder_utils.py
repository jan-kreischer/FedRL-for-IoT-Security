import torch 
from torch import nn

def initial_autoencoder_architecture():
    return nn.Sequential(
        nn.Linear(46, 23),
        nn.BatchNorm1d(23),
        nn.GELU(),
        nn.Linear(23, 11),
        nn.GELU(),
        nn.Linear(11, 23),
        nn.BatchNorm1d(23),
        nn.GELU(),
        nn.Linear(23, 46),
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

