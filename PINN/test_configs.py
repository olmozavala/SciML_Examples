from pinn_dash_app import _run_training, _read_loss_history

def test_config(strategy, learning_type, adaptive):
    print(f"Testing {strategy}, {learning_type}, {adaptive}")
    method_id = f"{strategy}_{learning_type}_{'_'.join(adaptive) if adaptive else 'none'}"
    
    _run_training(
        method_id=method_id,
        equation="heat",
        n_x=10,
        n_t=10,
        epochs=10,
        lr=0.01,
        strategy=strategy,
        learning_type=learning_type,
        adaptive_toggles=adaptive
    )
    res = _read_loss_history(method_id)
    if res.get("status") == "error":
        print(f"ERROR in {strategy}, {learning_type}, {adaptive}: {res.get('error')}")
    else:
        print(f"SUCCESS in {strategy}, {learning_type}, {adaptive}")

configs = [
    ("standard", "curriculum", []),
    ("vpinn", "standard", []),
    ("cpinn", "standard", []),
    ("standard", "standard", ["weights"]),
    ("standard", "standard", ["resampling"]),
]

for c in configs:
    test_config(*c)
