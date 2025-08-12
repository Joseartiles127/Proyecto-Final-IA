# Nombre: Jose Alberto Artiles
# Matrícula: 23-MISN-2-013
# Archivo: model.py
# Descripción: MLP simple para clasificar gestos a partir de landmarks (42 valores)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureNet(nn.Module):
    def __init__(self, input_dim=42, hidden=128, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.bn2 = nn.BatchNorm1d(hidden//2)
        self.fc3 = nn.Linear(hidden//2, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
