import random
from abc import abstractmethod, ABCMeta
from copy import deepcopy
from math import nan

import numpy as np
import torch
from torch.utils.data import DataLoader


class Participant(metaclass=ABCMeta):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        data_train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).type(torch.float),
            torch.from_numpy(train_y).type(y_type)
        )
        self.data_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        # shuffling is not really necessary
        # and batch size should also be irrelevant, can be set to whatever fits in memory
        data_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).type(torch.float),
            torch.from_numpy(valid_y).type(y_type)
        )
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size_valid, shuffle=True)
        self.validation_losses = []

        self.model = None
        self.threshold = nan

    @abstractmethod
    def train(self, optimizer, loss_function, num_local_epochs):
        pass

    def get_model(self):
        return self.model

    def set_model(self, model: torch.nn.Module):
        self.model = model


class MLPParticipant(Participant):

    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        if self.model is None:
            raise ValueError("No model set on participant!")

        epoch_losses = []
        validation_losses = []
        for le in range(num_local_epochs):
            self.model.train()
            current_losses = []
            for batch_idx, (x, y) in enumerate(self.data_loader):
                x, y = x, y  # x.cuda(), y.cuda()
                optimizer.zero_grad()
                model_predictions = self.model(x)
                loss = loss_function(model_predictions, y)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            epoch_losses.append(sum(current_losses) / len(current_losses))
            # print(f'Training Loss in epoch {le + 1}: {epoch_losses[le]}')

            with torch.no_grad():
                self.model.eval()
                current_losses = []
                for batch_idx, (x, y) in enumerate(self.valid_loader):
                    x, y = x, y  # x.cuda(), y.cuda()
                    model_predictions = self.model(x)
                    loss = loss_function(model_predictions, y)
                    current_losses.append(loss.item())
                validation_losses.append(sum(current_losses) / len(current_losses))
                # print(f'Validation Loss in epoch {le + 1}: {sum(current_losses) / len(current_losses)}')

            self.validation_losses = self.validation_losses + validation_losses

            if validation_losses[le] < 1e-4 or (le > 0 and (validation_losses[le] - validation_losses[le - 1]) > 1e-4):
                # print(f"Early stopping criterion reached in epoch {le + 1}")
                return


# Goal: raise alarm on every sample - TNR -> 0%
class BenignLabelFlipAdversary(MLPParticipant):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        train_y[train_y == 0] = 1
        valid_y[valid_y == 0] = 1
        super().__init__(train_x, train_y, valid_x, valid_y, batch_size, batch_size_valid, y_type)


# Goal: raise alarm on none of the samples - TPR -> 0%
class AttackLabelFlipAdversary(MLPParticipant):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        train_y[train_y == 1] = 0
        valid_y[valid_y == 1] = 0
        super().__init__(train_x, train_y, valid_x, valid_y, batch_size, batch_size_valid, y_type)


# Goal: completely destroy model - drive accuracy to 0%
class AllLabelFlipAdversary(MLPParticipant):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        train_y = (train_y != 1).astype(np.longlong)
        valid_y = (valid_y != 1).astype(np.longlong)
        super().__init__(train_x, train_y, valid_x, valid_y, batch_size, batch_size_valid, y_type)


# robust AE methods for adversarial part: https://ieeexplore.ieee.org/document/9099561
class AutoEncoderParticipant(Participant):

    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        if self.model is None:
            raise ValueError("No model set on participant!")
        epoch_losses = []
        for le in range(num_local_epochs):
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


class ModelCancelAdversary(AutoEncoderParticipant):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray, n_honest: int, n_malicious: int,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        super().__init__(train_x, train_y, valid_x, valid_y, batch_size, batch_size_valid, y_type)
        self.n_honest = n_honest
        self.n_malicious = n_malicious

    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        # before local training, the participant model corresponds to the old global model / except initialization
        old_global_model = deepcopy(self.get_model())

        factor = - 1. * self.n_honest / self.n_malicious
        with torch.no_grad():
            new_weights = {}
            for key, original_param in old_global_model.state_dict().items():
                new_weights.update({key: original_param * (factor if "running_" not in key else 1)})
            self.model.load_state_dict(new_weights)

    # simple overstatement
    def determine_threshold(self) -> float:
        return random.uniform(1e6, 1e+9)

# if threshold is selected like in the normal AutoEnc. Participant sending random weights is not an efficient attack
# a model with 100% malicious participants still recognizes some behaviors with 100% accuracy without attacking the threshold
class RandomWeightAdversary(AutoEncoderParticipant):
    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        with torch.no_grad():
            state_dict = self.model.state_dict()
            new_dict = deepcopy(state_dict)
            for key in state_dict.keys():
                new_dict[key] = torch.randn(state_dict[key].size()) * 3
                if "running_" in key:
                    new_dict[key] = torch.abs(new_dict[key])
            self.model.load_state_dict(new_dict)

    def determine_threshold(self) -> float:
        # so a MSE of sqrt(68) would be being 1 off per feature
        # (which is much considering MinMax scaling)
        # However anomalies have proven to be way more off (due to the scaling).
        # Therefore this participant also overstates the threshold randomly
        # To be validated if it makes sense (basically combines threshold overstatement and random params)
        return random.uniform(1e6, 1e+9)


# Approx thresholds ranges of non-attacked device types:
# Ras-3: ~7.5-8
# Ras-4-4gb: ~1.7-1.9
# Ras-4-2gb: ~1.8-2.1

# Goal: make as many attacks as possible recognized as normals
# Side effect: normal samples recognized as normals across heterogeneous devices
class ExaggerateThresholdAdversary(AutoEncoderParticipant):

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
        # just way more than normal participants threshold
        return mses.mean() + 15 * mses.std()


# GOAL: make the model also raise alarm on normal samples
class UnderstateThresholdAdversary(AutoEncoderParticipant):

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
        # just way less than normal participants threshold
        # achieves goal with only one participant,
        return mses.mean() - 15 * mses.std()
