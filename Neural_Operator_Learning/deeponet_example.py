"""Simple GPU-only DeepONet example for a 1D Poisson operator."""
from __future__ import annotations

import torch

from operator_data import describe_dataset, make_dataset, set_seed
from operator_models import DeepONet, predict_deeponet, require_cuda_device, summarize_prediction
from training import train_deeponet


device = require_cuda_device()


@torch.no_grad()
def evaluate(model: DeepONet, test_data: dict[str, torch.Tensor]) -> None:
    """Evaluate on full test functions."""
    target = test_data["solution"].to(device)
    prediction = predict_deeponet(model, test_data, device)
    metrics = summarize_prediction(prediction, target)
    print(f"Test MSE: {metrics['mse']:.6e}")
    print(f"Relative L2 error: {metrics['relative_l2']:.6e}")
    print("Example target values:", target[0, :5].cpu().numpy())
    print("Example prediction  :", prediction[0, :5].cpu().numpy())


def main() -> None:
    """Train and evaluate a simple DeepONet."""
    set_seed(7)
    train_data = make_dataset(n_samples=128, n_points=64, n_modes=6, seed=7)
    test_data = make_dataset(n_samples=32, n_points=64, n_modes=6, seed=17)

    print(f"Using device: {device}")
    print("DeepONet training set:", describe_dataset(train_data))
    model = DeepONet(n_sensors=train_data["forcing"].shape[1]).to(device)
    history = train_deeponet(model, train_data, device=device, epochs=250, learning_rate=1e-3, batch_size=1024, log_interval=50)
    for entry in history:
        print(f"Epoch {int(entry['epoch']):4d} | train MSE: {entry['train_loss']:.6e}")
    evaluate(model, test_data)


if __name__ == "__main__":
    main()
