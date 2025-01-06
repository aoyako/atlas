import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def objective(train_loader, val_loader):
    def search(trial):
        n_layers = trial.suggest_int("n_layers", 1, 5)
        n_units = trial.suggest_int("n_units", 32, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = SimpleNN(16, n_layers, n_units, dropout_rate)

        early_stopper = EarlyStopping(patience=10)
        trainer = ModelTrainer(model, early_stopper=early_stopper, lr=learning_rate)

        trainer.train(train_loader, val_loader, epochs=1000)
        return trainer.validate(val_loader)
    return search

def optuna_best(study, train_loader, val_loader):
    best_params = study.best_params

    model = SimpleNN(16, best_params['n_layers'], best_params['n_units'], best_params['dropout_rate'])

    early_stopper = EarlyStopping(patience=10)
    trainer = ModelTrainer(model, early_stopper=early_stopper, lr=best_params['learning_rate'])

    trainer.train(train_loader, val_loader, epochs=1000)
    return model

class SimpleNN(nn.Module):
    def __init__(self, input_dim, n_units, n_layers, dropout_rate):
        super(SimpleNN, self).__init__()
        self.network = []

        for _ in range(n_layers):
            self.network.append(nn.Linear(input_dim, n_units))
            self.network.append(nn.BatchNorm1d(n_units))
            self.network.append(nn.ReLU())
            self.network.append(nn.Dropout(dropout_rate))
            input_dim = n_units
        
        self.network.append(nn.Linear(input_dim, 13))
        self.network.append(nn.Sigmoid())

        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, model, lr=0.001, early_stopper=None):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.train_hist = []
        self.val_hist = []
        self.early_stopper = early_stopper

    def train(self, train_loader, val_loader, epochs):
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            self.train_hist.append(epoch_loss / len(train_loader))
            self.val_hist.append(val_loss)

            if self.early_stopper and self.early_stopper(val_loss, self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break

            self.scheduler.step()

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), targets.float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for inputs, targets in test_loader:
                inputs, targets = inputs.float(), targets.float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

            print(f"Test Loss: {total_loss / len(test_loader):.4f}")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stop = False

    def __call__(self, val_loss, model, path='checkpoint.pth'):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            torch.save(model.state_dict(), path)
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True
        return self.early_stop