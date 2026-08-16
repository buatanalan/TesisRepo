import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json
from marl_spklu.rl.training import _fresh_sim
from marl_spklu.rl.forecaster import collect_forecast_dataset, MLPForecaster
from marl_spklu.agents.greedy_agent import GreedyAgent

sim = _fresh_sim("scenario_dataset.json")
X, y = collect_forecast_dataset(sim, min(2880, sim.max_steps), agent=GreedyAgent())
X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
print("n_pairs:", X.shape[0])

model = MLPForecaster(X.shape[1], 64)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
X_t = torch.tensor(X); y_t = torch.tensor(y)
ds = torch.utils.data.TensorDataset(X_t, y_t)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

history = []
for epoch in range(10):
    model.train()
    epoch_losses = []
    for bx, by in loader:
        opt.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        full_pred = model(X_t).numpy()
    mae = float(np.mean(np.abs(full_pred - y)))
    mse_full = float(np.mean((full_pred - y) ** 2))
    history.append({"epoch": epoch, "mean_batch_loss": float(np.mean(epoch_losses)),
                    "full_dataset_mse": mse_full, "full_dataset_mae": mae})
    print(f"epoch {epoch}: mean_batch_loss(MSE)={np.mean(epoch_losses):.2f}  "
          f"full_MSE={mse_full:.2f}  full_MAE={mae:.2f}")

with open("check_fit_loss_history.json", "w") as f:
    json.dump(history, f, indent=2)
