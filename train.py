# Nombre: Jose Alberto Artiles
# Matrícula: 23-MISN-2-013
# Archivo: train.py
# Descripción: Entrenamiento básico del GestureNet con datos locales dataset_gestos.npz

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import GestureNet
import torch.optim as optim
import torch.nn.functional as F
import os

DATA_PATH = "dataset_gestos.npz"
OUT_PATH = "modelo/gesture_model.pt"
os.makedirs("modelo", exist_ok=True)

def load_data(path):
    data = np.load(path)
    X = data['X'].astype(np.float32)  # shape (N,42)
    y = data['y'].astype(np.int64)    # shape (N,)
    return X, y

def train():
    X, y = load_data(DATA_PATH)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    n_classes = len(np.unique(y))
    model = GestureNet(input_dim=X.shape[1], n_classes=n_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), OUT_PATH)
    print("Modelo guardado en", OUT_PATH)

if __name__ == "__main__":
    train()
