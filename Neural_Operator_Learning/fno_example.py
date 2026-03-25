"""Simple GPU-only 1D Fourier Neural Operator example for a Poisson operator."""
from __future__ import annotations

import torch

from operator_data import describe_dataset, make_dataset, set_seed
from operator_models import FNO1d, predict_fno, require_cuda_device, summarize_prediction, train_fno


device = require_cuda_device()


@torch.no_grad()
def evaluate(model: FNO1d, test_data: dict[str, torch.Tensor]) -> None:
    """Evaluate on full test functions."""
    target = test_data["solution"].to(device)
    prediction = predict_fno(model, test_data, device)
    metrics = summarize_prediction(prediction, target)
    print(f"Test MSE: {metrics['mse']:.6e}")
    print(f"Relative L2 error: {metrics['relative_l2']:.6e}")
    print("Example target values:", target[0, :5].cpu().numpy())
    print("Example prediction  :", prediction[0, :5].cpu().numpy())


def main() -> None:
    """Train and evaluate a simple FNO."""
    set_seed(7)
    train_data = make_dataset(n_samples=512, n_points=64, n_modes=6, seed=7)
    test_data = make_dataset(n_samples=64, n_points=64, n_modes=6, seed=17)

    print(f"Using device: {device}")
    print("FNO training set:", describe_dataset(train_data))
    model = FNO1d(input_dim=2, width=32, modes=12, depth=4).to(device)
    history = train_fno(model, train_data, device=device, epochs=300, learning_rate=1e-3, batch_size=32, log_interval=75)
    for entry in history:
        print(f"Epoch {int(entry['epoch']):4d} | train MSE: {entry['train_loss']:.6e}")
    evaluate(model, test_data)


if __name__ == "__main__":
    main()
