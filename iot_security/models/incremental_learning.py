"""
Incremental Learning Module
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EWC:
    def __init__(self, model: nn.Module, dataloader, device, n_samples: int = 500):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader, n_samples)

    def _compute_fisher(self, dataloader, n_samples):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for X_batch, y_batch in dataloader:
            if count >= n_samples:
                break
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.model.zero_grad()
            logits = self.model(X_batch)
            log_probs = torch.log_softmax(logits, dim=1)
            preds = logits.argmax(1)
            loss = log_probs[range(len(preds)), preds].mean()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += len(X_batch)
        fisher = {n: f / count for n, f in fisher.items()}
        return fisher

    def penalty(self, current_model: nn.Module):
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in current_model.named_parameters():
            if p.requires_grad and n in self.fisher:
                penalty += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return penalty

def incremental_train(model, X_new, y_new, device, ewc: EWC = None, ewc_lambda: float = 400.0, epochs: int = 5, lr: float = 5e-4, batch_size: int = 128, progress_callback=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tx = torch.tensor(X_new, dtype=torch.float32)
    ty = torch.tensor(y_new, dtype=torch.long)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            task_loss = criterion(logits, y_batch)
            ewc_loss = torch.tensor(0.0, device=device)
            if ewc is not None:
                ewc_loss = ewc_lambda * ewc.penalty(model)
            loss = task_loss + ewc_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        avg_loss = total_loss / total
        avg_acc  = correct / total
        history.append({'epoch': epoch, 'loss': avg_loss, 'acc': avg_acc})
        if progress_callback:
            progress_callback(epoch, avg_loss, avg_acc)
    return history
