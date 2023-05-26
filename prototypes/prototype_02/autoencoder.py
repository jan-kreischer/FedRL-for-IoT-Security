from torch import nn
import torch
import numpy as np
from src.custom_types import Behavior
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score
from tabulate import tabulate

class AutoEncoder(torch.nn.Module):
    

    def __init__(self, model, X_valid, X_test, y_test, evaluation_data, n_std=0.5, activation_function=torch.nn.ReLU(), batch_size: int = 64, verbose=False):

        super().__init__()
        
        validation_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_valid).type(torch.float),

        )
        self.validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True, drop_last=True)

        self.X_test = X_test
        self.y_test = y_test
        
        self.evaluation_data = evaluation_data
        self.n_std = n_std
        
        n_features = X_test.shape[1]
        
        print("using 64 x 16")
        self.model = nn.Sequential(
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
        self.threshold = None
        self.loss_mean = None
        self.loss_standard_deviation = None
        
        self.verbose = verbose

        
    def forward(self, X):
        return self.model(X)
    
    
    def pretrain(self, X_train, optimizer=torch.optim.SGD, loss_function=torch.nn.MSELoss(reduction='sum'), num_epochs: int = 15, batch_size=64, verbose=False):
        
        training_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).type(torch.float),
        )
        training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        epoch_losses = []
        #for e in tqdm(range(num_epochs), unit="epoch", leave=False):
        for e in range(num_epochs):
            self.train()
            current_losses = []
            for batch_index, (inputs,) in enumerate(training_data_loader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_function(inputs, outputs)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            
            epoch_losses.append(np.average(current_losses))
            if verbose:
                print(f'Training Loss in epoch {e + 1}: {epoch_losses[e]}')
            
        self.analyze_loss()

    '''
    This function uses normal data samles 
    after training the autoencoder to determine
    values that can be considered normal
    for the reconstruction loss based on normal samples
    '''
    def analyze_loss(self):
        losses = []
        
        self.eval() 
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_index, (inputs,) in enumerate(self.validation_data_loader):
                outputs = self.forward(inputs)
                loss = loss_function(inputs, outputs)
                losses.append(loss.item())
        
        losses = np.array(losses)

        self.loss_mean = losses.mean()
        self.loss_standard_deviation = losses.std()

        
    def predict(self, x, n_std=None):
        
        if n_std==None:
            n_std = self.n_std
        else:
            n_std = self.n_std
            
        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(x).type(torch.float32)
        )
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        all_predictions = torch.tensor([])  # .cuda()

        self.eval()
        with torch.no_grad():
            ae_loss = torch.nn.MSELoss(reduction="sum")
            for idx, (batch_x,) in enumerate(test_data_loader):
                model_predictions = self.forward(batch_x)
                model_predictions = ae_loss(model_predictions, batch_x).unsqueeze(0)  # unsqueeze as batch_size set to 1
                all_predictions = torch.cat((all_predictions, model_predictions))

        threshold = self.loss_mean + n_std * self.loss_standard_deviation
        all_predictions = (all_predictions > threshold).type(torch.long)
        return all_predictions.flatten()
    
    
    def predict_deviation(self, x):
        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(x).type(torch.float32)
        )
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        prediction_errors = torch.tensor([])
        loss_function = torch.nn.MSELoss(reduction="sum")
        
        self.eval()
        with torch.no_grad():
            
            for batch_index, (inputs,) in enumerate(test_data_loader):
                prediction = self.forward(inputs)
                prediction_error = loss_function(inputs, prediction).unsqueeze(0)  # unsqueeze as batch_size set to 1
                prediction_errors = torch.cat((prediction_errors, prediction_error))

        return prediction_errors
    
    
    def score(self):
        n_std, accuracy = self.accuracy_score(None, None)
        if self.verbose:
            print(f"Highest validation accuracy achieved {accuracy:.2f} with n_std={n_std}")
            self.evaluate(n_std)
        return accuracy
    
    
    def accuracy_score(self, X, y):
        #if not self.threshold:
        #loss_mean, loss_standard_deviation = self.analyze_loss(X)
        #n_stds = np.arange(0.1, 3, 0.1)
        if self.loss_mean == None or self.loss_standard_deviation == None:
              #print("accuracy_score_optimized > accurcy_loss()")
              self.analyze_loss()
    
        best_accuracy = 0
        best_n_std = 0
        #accuracies = []
        y_dev = self.predict_deviation((self.X_test).astype(np.float32))
        for n_std in self.n_stds:
            y_true = self.y_test
            threshold = self.loss_mean + n_std * self.loss_standard_deviation
            y_pred = (y_dev > threshold).type(torch.long).detach().cpu().numpy()
            
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_n_std = n_std
            #if self.verbose:
            #    print(f"n_std {n_std:.2f} -> accuracy: {accuracy}")

        return best_n_std, best_accuracy
    
    
    def evaluate(self, n_std=None, tablefmt='pipe'):
        results = []
        labels= [0,1]
        pos_label = 1
        
        if n_std==None:
            n_std = self.n_std
        else:
            n_std = self.n_std
        
        y_true_total = np.empty([0])
        y_pred_total = np.empty([0])
        for behavior, data in self.evaluation_data.items():
            y_true = np.array([0 if behavior == Behavior.NORMAL else 1] * len(data)).astype(int)
            y_true_total = np.concatenate((y_true_total, y_true))

            y_pred = self.predict(data[:, :-1].astype(np.float32), n_std=n_std)
            y_pred_total = np.concatenate((y_pred_total, y_pred))

            accuracy = accuracy_score(y_true, y_pred)

            n_samples = len(y_true)
            results.append([behavior.name.replace("_", "\_"), f'{(100 * accuracy):.2f}\%', '\\notCalculated', '\\notCalculated', '\\notCalculated', str(n_samples)])

        accuracy = accuracy_score(y_true_total, y_pred_total)
        precision = precision_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        recall = recall_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        f1 = f1_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        n_samples = len(y_true_total)
        results.append(["GLOBAL", f'{(100 * accuracy):.2f}\%', f'{(100 * precision):.2f}\%', f'{(100 * recall):.2f}\%', f'{(100 * f1):.2f}\%', n_samples])
        print("-----------")
        print(tabulate(results, headers=["Behavior", "Accuracy", "Precision", "Recall", "F1-Score", "\#Samples"], tablefmt=tablefmt)) 