"""
Neural Ordinary Differential Equations
PyTorch
"""




"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _pick_device()

# Loto 7/39 ograničenja po poziciji (normalizacija u [0,1] po koloni)
_min_pos = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
_max_pos = np.array([33, 34, 35, 36, 37, 38, 39], dtype=np.float32)
_ranges = (_max_pos - _min_pos).reshape(1, 7)


def _norm_loto(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) - _min_pos) / _ranges


def _denorm_loto(arr: np.ndarray) -> np.ndarray:
    return arr * _ranges + _min_pos


# Učitaj CSV fajl
df = pd.read_csv("/data/loto7_4584_k23.csv", header=None)
df = df.iloc[:, :7]
data = df.values.astype(np.float32)

# X - sve kombinacije osim poslednje
X = data[:-1]
# y - sve kombinacije osim prve (pomereno za 1)
y = data[1:]

Xn = _norm_loto(X)
yn = _norm_loto(y)

# v2: hronološki train (prvih 85%) / val (zadnjih 15%)
_n = len(Xn)
_split = int(_n * 0.85)
X_tr, X_va = Xn[:_split], Xn[_split:]
y_tr, y_va = yn[:_split], yn[_split:]

# Torch tensori
X_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_tr, dtype=torch.float32, device=device)
X_val_t = torch.tensor(X_va, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_va, dtype=torch.float32, device=device)

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
model = NeuralODERegressor().to(device)
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
    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)
    loss.backward()
    optimizer.step()

    # print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print()

model.eval()
with torch.no_grad():
    train_loss_f = criterion(model(X_tensor), y_tensor).item()
    val_loss_f = criterion(model(X_val_t), y_val_t).item()

if epoch == 299:
    print(f"Epoch {epoch}: train MSE = {train_loss_f:.6f}  |  val MSE = {val_loss_f:.6f}")
print()
"""
Epoch 299: train MSE = 0.026277  |  val MSE = 0.026414
"""


# Predikcija sledeće kombinacije (v2: 7 pozicija u redosledu, clip po koloni; bez sort/random dopune)
model.eval()
with torch.no_grad():
    last_n = _norm_loto(data[-1].reshape(1, -1))
    last_input = torch.tensor(last_n, dtype=torch.float32, device=device)
    pred = model(last_input)
    pred_cpu = pred.squeeze().detach().cpu().numpy().reshape(1, -1)
    pred_denorm = _denorm_loto(pred_cpu).ravel()
    pred_int = np.round(pred_denorm).astype(int)
    for _j in range(7):
        pred_int[_j] = int(np.clip(pred_int[_j], _min_pos[_j], _max_pos[_j]))

    print("\n🎯 Predikcija sledeće loto kombinacije (pozicije 1–7):", pred_int.tolist())
    print("   ", " ".join(str(int(x)) for x in pred_int))
print()
"""
🎯 Predikcija sledeće loto kombinacije (pozicije 1–7): [5, 10, x, 19, y, z, 35]
    5 10 x 19 y z 35
"""
