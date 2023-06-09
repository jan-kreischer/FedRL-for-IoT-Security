from torch import nn
import torch
import numpy as np


class AutoEncoderInterpreter():

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64):
        data_train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).type(torch.float),
            torch.from_numpy(train_y).type(torch.float)
        )
        self.data_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        data_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).type(torch.float),
            torch.from_numpy(valid_y).type(torch.float)
        )
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size_valid, shuffle=True)
        self.validation_losses = []

        self.model = self.auto_encoder_model(train_x.shape[1])
        self.threshold = np.nan


    def auto_encoder_model(in_features: int, hidden_size: int = 32):
        return nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_features),
            nn.GELU()
        )


    def train(self, optimizer, loss_function, num_epochs: int = 15):
        if self.model is None:
            raise ValueError("No model set!")
        epoch_losses = []
        for le in range(num_epochs):
            self.model.train()
            current_losses = []
            for batch_idx, (x, _) in enumerate(self.data_loader):
                x = x  # x.cuda()
                optimizer.zero_grad()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            epoch_losses.append(sum(current_losses) / len(current_losses))
            # print(f'Training Loss in epoch {le + 1}: {epoch_losses[le]}')

    def determine_threshold(self) -> float:
        mses = []
        self.model.eval()
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_idx, (x, _) in enumerate(self.valid_loader):
                x = x  # x.cuda()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                mses.append(loss.item())
        mses = np.array(mses)
        return mses.mean() + 3 * mses.std()