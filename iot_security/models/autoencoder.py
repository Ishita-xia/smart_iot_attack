"""
Autoencoder
===========
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 39, bottleneck_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error(self, x):
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)

def train_autoencoder(model, X_benign: np.ndarray, device, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, progress_callback=None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tx = torch.tensor(X_benign, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tx), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(X_benign)
        history.append(avg_loss)
        print(f"  AE Epoch {epoch:02d}/{epochs}  loss={avg_loss:.6f}")
        if progress_callback:
            progress_callback(epoch, avg_loss)
    return history

def compute_threshold(model, X_benign: np.ndarray, device, percentile: float = 95):
    model.eval()
    tx = torch.tensor(X_benign, dtype=torch.float32).to(device)
    with torch.no_grad():
        errors = model.reconstruction_error(tx).cpu().numpy()
    threshold = np.percentile(errors, percentile)
    return threshold, errors

def detect_anomalies(model, X: np.ndarray, threshold: float, device):
    model.eval()
    tx = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        errors = model.reconstruction_error(tx).cpu().numpy()
    return errors > threshold, errors
