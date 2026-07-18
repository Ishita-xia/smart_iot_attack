"""
Adversarial Attack Module
"""
import torch
import torch.nn as nn
import numpy as np

def fgsm_attack(model, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1, device: str = 'cpu'):
    model.eval()
    tx = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    ty = torch.tensor(y, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()
    logits = model(tx)
    loss = criterion(logits, ty)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * tx.grad.data.sign()
    x_adv = tx.detach() + perturbation
    x_adv = x_adv.cpu().numpy()
    orig_acc  = _accuracy(model, X, y, device)
    adv_acc   = _accuracy(model, x_adv, y, device)
    return x_adv, orig_acc, adv_acc

def pgd_attack(model, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1, alpha: float = 0.01, num_steps: int = 10, device: str = 'cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tx_orig = torch.tensor(X, dtype=torch.float32).to(device)
    ty      = torch.tensor(y, dtype=torch.long).to(device)
    x_adv = tx_orig + torch.empty_like(tx_orig).uniform_(-epsilon, epsilon)
    for step in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss   = criterion(logits, ty)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - tx_orig, -epsilon, epsilon)
            x_adv = (tx_orig + delta).detach()
    x_adv_np  = x_adv.cpu().numpy()
    orig_acc  = _accuracy(model, X, y, device)
    adv_acc   = _accuracy(model, x_adv_np, y, device)
    return x_adv_np, orig_acc, adv_acc

def _accuracy(model, X: np.ndarray, y: np.ndarray, device: str):
    model.eval()
    tx = torch.tensor(X, dtype=torch.float32).to(device)
    ty = torch.tensor(y, dtype=torch.long)
    with torch.no_grad():
        preds = model(tx).argmax(1).cpu()
    return (preds == ty).float().mean().item()
