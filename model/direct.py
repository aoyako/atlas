import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 13),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, model, lr=0.001, early_stopper=None):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
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