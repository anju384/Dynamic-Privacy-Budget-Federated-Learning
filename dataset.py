import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_clients_data(num_clients=5):
    cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = np.random.randn(1000, 29)
    y = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

    client_loaders = []
    chunk_size = len(X) // num_clients

    for i in range(num_clients):
        X_c = torch.tensor(X[i*chunk_size:(i+1)*chunk_size], dtype=torch.float32)
        y_c = torch.tensor(y[i*chunk_size:(i+1)*chunk_size], dtype=torch.long)

        loader = DataLoader(TensorDataset(X_c, y_c), batch_size=32, shuffle=True)
        client_loaders.append(loader)

    return client_loaders