from torch import nn
import torch
import numpy as np
from tqdm import tqdm


def auto_encoder_model(in_features: int, hidden_size: int = 8):
    return nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, int(hidden_size / 2)),
        nn.GELU(),
        nn.Linear(int(hidden_size / 2), hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, in_features),
        nn.GELU()
    )


class AutoEncoder():

    def __init__(self, train_x: np.ndarray,
                 valid_x: np.ndarray,
                 batch_size: int = 64, batch_size_valid=1):

        data_train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).type(torch.float),
            #torch.from_numpy(train_y).type(torch.float)
        )
        self.data_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        data_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).type(torch.float),
            #torch.from_numpy(valid_y).type(torch.float)
        )
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size_valid, shuffle=True)
        self.validation_losses = []

        self.model = auto_encoder_model(in_features=train_x.shape[1])
        self.threshold = np.nan

    def get_model(self):
        return self.model

    def train(self, optimizer, loss_function=torch.nn.MSELoss(reduction='sum'), num_epochs: int = 15):
        if self.model is None:
            raise ValueError("No model set!")

        epoch_losses = []
        # for e in tqdm(range(num_epochs), unit="epoch", leave=False):
        for e in range(num_epochs):
            self.model.train()
            current_losses = []
            for batch_idx, (x,) in enumerate(self.data_loader):
                x = x  # x.cuda()
                optimizer.zero_grad()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            epoch_losses.append(sum(current_losses) / len(current_losses))
            print(f'Training Loss in epoch {e + 1}: {epoch_losses[e]}')

    def determine_threshold(self) -> float:
        mses = []
        self.model.eval()
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_idx, (x,) in enumerate(self.valid_loader):
                x = x  # x.cuda()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                mses.append(loss.item())
        mses = np.array(mses)
        self.threshold = mses.mean() + 1 * mses.std()
        return self.threshold

    def save_model(self, dir="", model_name="ae_model.pth"):
        path = f"{dir}trained_models/{model_name}"
        print("save model to: " + path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold
        }, path)




class AutoEncoderInterpreter():
    def __init__(self, state_dict, threshold, in_features=15, hidden_size=8):
        self.model = auto_encoder_model(in_features=in_features, hidden_size=hidden_size)
        self.model.load_state_dict(state_dict)
        self.threshold = threshold

    def predict(self, x):
        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(x).type(torch.float)
        )
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        all_predictions = torch.tensor([])  # .cuda()

        self.model.eval()
        with torch.no_grad():
            ae_loss = torch.nn.MSELoss(reduction="sum")
            for idx, (batch_x,) in enumerate(data_loader):
                batch_x = batch_x  # .cuda()
                model_predictions = self.model(batch_x)

                model_predictions = ae_loss(model_predictions, batch_x).unsqueeze(0)  # unsqueeze as batch_size set to 1
                all_predictions = torch.cat((all_predictions, model_predictions))

        # all_predictions = all_predictions.tolist()
        all_predictions = (all_predictions > self.threshold).type(torch.long)
        return all_predictions.flatten()
