"""Export the actual Gaussian network, edge curves, and reproducible CPU demos."""

import json
from pathlib import Path

import torch

from kan_network import KAN
from kan_trainer import KANTrainer


def target_1d(x):
    return torch.sin(3 * x[:, :1]) + 0.3 * torch.cos(10 * x[:, :1])


def target_2d(x):
    return torch.sin(x[:, :1]) * torch.exp(-x[:, 1:2] ** 2)


def target_complex(x):
    return torch.sin(x[:, :1] * x[:, 1:2]) + 0.5 * torch.tanh(x[:, :1] - x[:, 1:2])


TASKS = {
    "1d_sine_wave": ("1D Sine Wave", "sin(3x) + 0.3*cos(10x)", 1, target_1d),
    "2d_gaussian": ("2D Gaussian Wave", "sin(x) * exp(-y²)", 2, target_2d),
    "2d_complex": ("2D Interaction", "sin(x*y) + 0.5*tanh(x-y)", 2, target_complex),
}

MODEL_CONFIGS = [
    ("model_1d", "1d_sine_wave", [1, 10, 1], 5, 1000, 100, 64, 0.01),
    ("model_2d", "2d_gaussian", [2, 15, 1], 5, 1500, 150, 128, 0.01),
    ("model_complex", "2d_complex", [2, 20, 15, 10, 1], 7, 2000, 200, 128, 0.005),
]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def export_kan_for_visualization(model, trainer=None, save_path="web/data/model.json", target_id=None):
    """Serialize all forward-pass parameters; sampled curves are for display only."""
    export_data = {
        "metadata": {
            "architecture": model.layers_hidden,
            "grid_size": model.grid_size,
            "basis": "gaussian_rbf",
            "num_layers": len(model.layers),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "target_id": target_id,
        },
        "layers": [],
        "training_history": trainer.training_history if trainer else None,
    }
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            x_fine = torch.linspace(-2, 2, 200, device=layer.grid.device, dtype=layer.grid.dtype)
            # Each output/input entry is independent of other input coordinates.
            x_inputs = x_fine[:, None].expand(-1, layer.in_features)
            edges = layer.edge_contributions(x_inputs).cpu()
            curves = [
                {"input_idx": in_idx, "output_idx": out_idx,
                 "x_values": x_fine.cpu().tolist(), "y_values": edges[:, out_idx, in_idx].tolist()}
                for out_idx in range(layer.out_features)
                for in_idx in range(layer.in_features)
            ]
            export_data["layers"].append({
                "layer_index": i,
                "input_features": layer.in_features,
                "output_features": layer.out_features,
                "grid_points": layer.grid.cpu().tolist(),
                "grid_range": list(layer.grid_range),
                "basis_sigma": layer.basis_sigma,
                "rbf_coefficients": layer.rbf_weight.detach().cpu().tolist(),
                "rbf_scalers": layer.rbf_scaler.detach().cpu().tolist() if layer.rbf_scaler is not None else None,
                "base_weights": layer.base_weight.detach().cpu().tolist(),
                "base_activation": layer.base_activation_name,
                "scale_base": layer.scale_base,
                "scale_rbf": layer.scale_rbf,
                "edge_evaluations": curves,
            })
    save_json(save_path, export_data)
    return export_data


def create_sample_datasets_for_web(save_path="web/data/datasets.json"):
    datasets = {}
    for task_id, (name, description, input_dim, target) in TASKS.items():
        if input_dim == 1:
            inputs = torch.linspace(-2, 2, 100).unsqueeze(1)
        else:
            axis = torch.linspace(-2, 2, 30)
            inputs = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
        outputs = target(inputs)
        datasets[task_id] = {
            "target_id": task_id, "name": name, "description": description,
            "input_dim": input_dim, "function": description,
            "samples": [{"input": x.tolist(), "output": float(y[0])} for x, y in zip(inputs, outputs)],
        }
    save_json(save_path, datasets)
    return datasets


def train_and_export_sample_models():
    """Regenerate demos from fixed seeds; report MSE on a separate fixed grid."""
    torch.set_num_threads(1)
    seed = 1729
    results = {"seed": seed, "device": "cpu", "torch_version": torch.__version__, "models": {}}
    for offset, (name, task_id, architecture, grid_size, n_samples, epochs, batch_size, lr) in enumerate(MODEL_CONFIGS):
        torch.manual_seed(seed + offset)
        target = TASKS[task_id][3]
        model = KAN(architecture, grid_size=grid_size)
        trainer = KANTrainer(model, device=torch.device("cpu"), optimizer_name="Adam", lr=lr)
        inputs, outputs = trainer.create_dataset(target, n_samples=n_samples, input_dim=architecture[0], x_range=(-2, 2))
        if architecture[0] == 1:
            evaluation_inputs = torch.linspace(-2, 2, 201).unsqueeze(1)
        else:
            axis = torch.linspace(-2, 2, 31)
            evaluation_inputs = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
        evaluation_targets = target(evaluation_inputs)
        before = trainer.evaluate(evaluation_inputs, evaluation_targets)
        trainer.train(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=False)
        after = trainer.evaluate(evaluation_inputs, evaluation_targets)
        export_kan_for_visualization(model, trainer, f"web/data/{name}.json", target_id=task_id)
        results["models"][name] = {
            "seed": seed + offset, "target_id": task_id, "architecture": architecture,
            "grid_size": grid_size, "training_samples": n_samples, "evaluation_samples": len(evaluation_inputs),
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "initial_grid_mse": before, "final_grid_mse": after,
        }
        print(f"{name}: evaluation grid MSE {before:.8g} -> {after:.8g}")
    create_sample_datasets_for_web()
    save_json("results/demo_training.json", results)
    return results


if __name__ == "__main__":
    train_and_export_sample_models()
