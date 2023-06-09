from torch import nn

def auto_encoder_model(in_features: int, hidden_size: int = 32):
    return nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, in_features),
        nn.GELU()
    )