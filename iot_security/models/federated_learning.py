"""
Federated Learning Simulation
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FederatedClient:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, batch_size: int = 128, local_epochs: int = 3, lr: float = 1e-3):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = lr
        tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        self.loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)
        self.n_samples = len(X)

    def local_train(self, global_model, device):
        model = copy.deepcopy(global_model).to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        return model.state_dict()

class FederatedServer:
    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device
        self.round_logs = []

    def aggregate(self, client_weights: list, client_sizes: list):
        total = sum(client_sizes)
        avg_state = copy.deepcopy(client_weights[0])
        for key in avg_state:
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for weights, size in zip(client_weights, client_sizes):
            factor = size / total
            for key in avg_state:
                avg_state[key] += weights[key].float() * factor
        self.global_model.load_state_dict(avg_state)

    def evaluate(self, test_loader):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                preds = self.global_model(X_batch).argmax(1).cpu()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return correct / total if total > 0 else 0.0

def split_data_for_clients(X: np.ndarray, y: np.ndarray, n_clients: int = 5, iid: bool = True):
    n = len(X)
    if iid:
        idx = np.random.permutation(n)
        splits = np.array_split(idx, n_clients)
        return [(X[s], y[s]) for s in splits]
    else:
        order = np.argsort(y)
        splits = np.array_split(order, n_clients)
        return [(X[s], y[s]) for s in splits]

def run_federated_simulation(global_model, X_tr, y_tr, X_te, y_te, n_clients=5, rounds=5, local_epochs=3, device='cpu', iid=True, round_callback=None):
    server = FederatedServer(global_model, device)
    tx = torch.tensor(X_te, dtype=torch.float32)
    ty = torch.tensor(y_te, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(tx, ty), batch_size=512)
    partitions = split_data_for_clients(X_tr, y_tr, n_clients, iid)
    clients = [FederatedClient(i, X_part, y_part, local_epochs=local_epochs) for i, (X_part, y_part) in enumerate(partitions)]
    results = []
    for rnd in range(1, rounds + 1):
        print(f"\n[SYNC] FL Round {rnd}/{rounds}")
        client_weights = []
        client_sizes   = []
        for client in clients:
            w = client.local_train(server.global_model, device)
            client_weights.append(w)
            client_sizes.append(client.n_samples)
        server.aggregate(client_weights, client_sizes)
        acc = server.evaluate(test_loader)
        results.append({'round': rnd, 'accuracy': acc})
        server.round_logs.append({'round': rnd, 'accuracy': acc})
        if round_callback:
            round_callback(rnd, acc)
    return results, server.global_model
