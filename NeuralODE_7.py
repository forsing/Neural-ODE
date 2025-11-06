"""
Neural Ordinary Differential Equations
PyTorch
"""


"""        
=== System Information ===
Python version                 3.11.13        
macOS Apple                    Tahos 
Apple                          M1
"""


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4506 izvlacenja
30.07.1985.- 04.11.2025.
"""



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import random
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Učitaj CSV fajl
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto7_4506_k87.csv", header=None)
data = df.values

# X - sve kombinacije osim poslednje
X = data[:-1]
# y - sve kombinacije osim prve (pomereno za 1)
y = data[1:]

# Normalizacija u [0, 1]
X = X / 39.0
y = y / 39.0

# Torch tensori
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Neural ODE dynamics
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),  # veći kapacitet
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7)   # izlaz je 7 brojeva
        )

    def forward(self, h):
        return self.net(h)

# Euler integrator
def euler_integrate(func, h0, t0, t1, n_steps):
    t = t0
    h = h0
    dt = (t1 - t0) / n_steps
    for _ in range(n_steps):
        h = h + func(h) * dt
        t += dt
    return h

# Neural ODE model
class NeuralODERegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.odefunc = ODEFunc()
        self.t0 = 0.0
        self.t1 = 1.0
        self.n_steps = 50

    def forward(self, x):
        h0 = x
        hT = euler_integrate(self.odefunc, h0, self.t0, self.t1, self.n_steps)
        return hT

# Model, optimizer, loss
model = NeuralODERegressor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Trenira model
print()
print("Treniranje modela ...")
"""
Treniranje modela ...
"""
print()

for epoch in range(300):
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)
    loss.backward()
    optimizer.step()

    # print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print()

if epoch==299:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
print()
"""
Epoch 299: Loss = 0.017727
"""


# Predikcija sledeće kombinacije
model.eval()
with torch.no_grad():
    last_input = torch.tensor(data[-1] / 39.0, dtype=torch.float32).unsqueeze(0)
    pred = model(last_input)
    pred_denorm = (pred.squeeze().numpy() * 39).round().astype(int)

    # Obezbedi validnu loto kombinaciju
    pred_denorm = np.clip(pred_denorm, 1, 39)          # ograniči na [1, 39]
    pred_unique = sorted(set(pred_denorm))             # ukloni duplikate
    while len(pred_unique) < 7:
        candidate = random.randint(1, 39)
        if candidate not in pred_unique:
            pred_unique.append(candidate)
    pred_unique.sort()

    print("\n🎯 Predikcija sledeće loto kombinacije:", pred_unique)
print()
"""
🎯 Predikcija sledeće loto kombinacije: [5, 10, 15, 20, 25, 30, 35]
"""
