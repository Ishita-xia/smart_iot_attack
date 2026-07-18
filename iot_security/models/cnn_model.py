"""
CNN Model (PyTorch)
====================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class CNN1D(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_out_size = 128 * (num_features // 4)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_loaders(X_tr, y_tr, X_te, y_te, batch_size=256):
    def make(X, y):
        tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)
    return make(X_tr, y_tr), make(X_te, y_te)

def train_cnn(model, train_loader, device, epochs=10, lr=1e-3, progress_callback=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    history = {'loss': [], 'acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc  = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        print(f"  Epoch {epoch:02d}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")
        if progress_callback:
            progress_callback(epoch, epoch_loss, epoch_acc)
    return history

def evaluate_cnn(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm
