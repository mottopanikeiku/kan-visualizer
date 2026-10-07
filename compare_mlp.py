"""Compare Gaussian-edge networks with dense MLPs on the demo tasks.

Each seed is a separate CPU job. Both families see exactly the same training
samples and minibatch order. Evaluation is only at the end, without selecting
checkpoints or hyperparameters from the evaluation grid.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics

import torch
from torch import nn

from export_for_web import MODEL_CONFIGS, TASKS, save_json
from kan_network import KAN

SEEDS = [1729, 1730, 1731, 1732, 1733]
# Dense widths are selected from parameter counts, not evaluation losses.
MLP_ARCHITECTURES = {
    "1d_sine_wave": [1, 53, 1],
    "2d_gaussian": [2, 90, 1],
    "2d_complex": [2, 65, 48, 33, 1],
}
CONFIGS = {
    task: dict(kan=architecture, mlp=MLP_ARCHITECTURES[task], grid_size=grid,
               samples=samples, epochs=epochs, batch_size=batch, lr=lr)
    for _, task, architecture, grid, samples, epochs, batch, lr in MODEL_CONFIGS
}
FAMILIES = ["gaussian_edge", "mlp_tanh", "mlp_silu"]


def make_model(config, family):
    if family == "gaussian_edge":
        return KAN(config["kan"], grid_size=config["grid_size"])
    activation = {"mlp_tanh": nn.Tanh, "mlp_silu": nn.SiLU}[family]
    layers = []
    widths = config["mlp"]
    for index, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(inputs, outputs))
        if index < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def evaluation_grid(dim):
    if dim == 1:
        return torch.linspace(-2, 2, 201).unsqueeze(1)
    axis = torch.linspace(-2, 2, 31)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def metrics(model, inputs, targets):
    residual = model(inputs) - targets
    return {"mse": residual.square().mean().item(),
            "mae": residual.abs().mean().item(),
            "max_absolute_error": residual.abs().max().item()}


def run_seed(seed, output_dir):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runs = []
    for task_id, config in CONFIGS.items():
        target = TASKS[task_id][3]
        # Dedicated generators separate data and batch order from initialization.
        data_generator = torch.Generator().manual_seed(seed)
        inputs = torch.rand(config["samples"], config["kan"][0], generator=data_generator) * 4 - 2
        targets = target(inputs)
        grid = evaluation_grid(inputs.shape[1])
        grid_targets = target(grid)
        batch_generator = torch.Generator().manual_seed(seed + 10000)
        orders = [torch.randperm(len(inputs), generator=batch_generator)
                  for _ in range(config["epochs"])]
        order_hash = tensor_hash(torch.stack(orders))
        for family in FAMILIES:
            torch.manual_seed(seed)
            model = make_model(config, family)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            initial = metrics(model, grid, grid_targets)
            steps = 0
            for order in orders:
                for indices in order.split(config["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(inputs[indices]) - targets[indices]).square().mean()
                    loss.backward()
                    optimizer.step()
                    steps += 1
            run = {
                "task_id": task_id, "seed": seed, "family": family,
                "architecture": config["kan"] if family == "gaussian_edge" else config["mlp"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "optimizer_steps": steps, "epochs": config["epochs"],
                "batch_size": config["batch_size"], "learning_rate": config["lr"],
                "training_samples": len(inputs), "evaluation_samples": len(grid),
                "training_inputs_sha256": tensor_hash(inputs),
                "minibatch_order_sha256": order_hash,
                "evaluation_inputs_sha256": tensor_hash(grid),
                "initial_grid": initial, "final_grid": metrics(model, grid, grid_targets),
                "final_training": metrics(model, inputs, targets),
            }
            if not all(math.isfinite(value) for value in run["final_grid"].values()):
                raise RuntimeError(f"Nonfinite result: {task_id}, {seed}, {family}")
            runs.append(run)
            print(f'{task_id} seed={seed} {family}: {run["parameters"]} parameters, '
                  f'{steps} steps, grid MSE={run["final_grid"]["mse"]:.8g}', flush=True)
    save_json(output_dir / f"seed_{seed}.json", {
        "seed": seed, "device": "cpu", "threads": torch.get_num_threads(),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "platform": platform.system(), "machine": platform.machine(), "runs": runs,
    })


def summarize(output_dir):
    raw_files = [output_dir / f"seed_{seed}.json" for seed in SEEDS]
    runs = [run for path in raw_files for run in json.loads(path.read_text())["runs"]]
    expected = {(task, seed, family) for task in CONFIGS for seed in SEEDS for family in FAMILIES}
    observed = [(run["task_id"], run["seed"], run["family"]) for run in runs]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Missing or duplicate task/seed/family results")
    for task in CONFIGS:
        for seed in SEEDS:
            paired = [run for run in runs if run["task_id"] == task and run["seed"] == seed]
            for field in ["training_inputs_sha256", "minibatch_order_sha256", "evaluation_inputs_sha256", "optimizer_steps"]:
                if len({run[field] for run in paired}) != 1:
                    raise ValueError(f"Unpaired {field}: {task}, {seed}")
    rows = []
    for task in CONFIGS:
        for family in FAMILIES:
            group = [run for run in runs if run["task_id"] == task and run["family"] == family]
            values = [run["final_grid"]["mse"] for run in group]
            rows.append({"task_id": task, "family": family,
                         "parameters": group[0]["parameters"], "optimizer_steps": group[0]["optimizer_steps"],
                         "median_grid_mse": statistics.median(values), "min_grid_mse": min(values),
                         "max_grid_mse": max(values), "mean_grid_mse": statistics.mean(values),
                         "median_grid_mae": statistics.median(run["final_grid"]["mae"] for run in group)})
    wins = {}
    for task in CONFIGS:
        task_runs = {(run["seed"], run["family"]): run for run in runs if run["task_id"] == task}
        wins[task] = {family: sum(task_runs[(seed, family)]["final_grid"]["mse"] <
                                  task_runs[(seed, "gaussian_edge")]["final_grid"]["mse"]
                                  for seed in SEEDS) for family in FAMILIES[1:]}
    summary = {"seeds": SEEDS, "runs": len(runs), "metric": "MSE on fixed separate evaluation grids; lower is better",
               "optimizer": "Adam, default betas and epsilon, no weight decay, fixed final step",
               "parameter_matching": "Same number of hidden layers; dense widths chosen for similar parameter counts, within 0.3%. Biases included in MLP counts; Gaussian scalers included in edge counts.",
               "rbf_grid_range": [-1, 1], "input_range": [-2, 2],
               "task_settings": CONFIGS,
               "environment_file": "results/mlp_comparison/environment.json",
               "raw_files": [str(path.as_posix()) for path in raw_files],
               "rows": rows, "paired_mlp_wins": wins}
    save_json(output_dir / "summary.json", summary)
    write_figure(rows, Path("docs/assets/mlp-comparison.svg"))
    print(json.dumps(summary, indent=2))


def write_figure(rows, path):
    # Dependency-free SVG; bars and ranges are derived directly from summary rows.
    colors = {"gaussian_edge": "#2366a8", "mlp_tanh": "#a54b00", "mlp_silu": "#657329"}
    labels = {"gaussian_edge": "Gaussian edge", "mlp_tanh": "MLP Tanh", "mlp_silu": "MLP SiLU"}
    values = [row[key] for row in rows for key in ["min_grid_mse", "max_grid_mse"]]
    lo, hi = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    def position(value):
        return 315 + (math.log10(value) - lo) / (hi - lo) * 395
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="560" viewBox="0 0 800 560" role="img" aria-labelledby="title desc">',
           '<title id="title">Parameter-matched MLP comparison</title>',
           '<desc id="desc">Final evaluation-grid MSE, five paired seeds. Dots are medians; lines span minimum to maximum. Lower is better. Values are in results/mlp_comparison/summary.json.</desc>',
           '<rect width="800" height="560" fill="white"/>',
           '<g font-family="sans-serif" font-size="15" fill="#17202b">',
           '<text x="24" y="30" font-size="22">Dense MLPs versus Gaussian-edge networks</text>',
           '<text x="24" y="57">Five paired seeds · median and min–max · lower MSE is better</text>']
    for power in range(lo, hi + 1):
        x = position(10 ** power)
        svg.extend([f'<path d="M{x:.2f} 90V478" stroke="#e0e5eb"/>',
                    f'<text x="{x:.2f}" y="504" text-anchor="middle">10^{power}</text>'])
    names = ["Sine wave", "Gaussian wave", "Interaction"]
    for index, row in enumerate(rows):
        y = 110 + index * 42
        if index % 3 == 0:
            svg.append(f'<text x="24" y="{y - 11}" font-weight="bold">{names[index // 3]}</text>')
        svg.extend([f'<text x="24" y="{y + 9}">{labels[row["family"]]} ({row["parameters"]} params)</text>',
                    f'<path d="M{position(row["min_grid_mse"]):.2f} {y}H{position(row["max_grid_mse"]):.2f}" stroke="{colors[row["family"]]}" stroke-width="3"/>',
                    f'<circle cx="{position(row["median_grid_mse"]):.2f}" cy="{y}" r="5" fill="{colors[row["family"]]}"/>',
                    f'<text x="735" y="{y + 5}" font-size="13">{row["median_grid_mse"]:.3g}</text>'])
    svg.extend(['<text x="510" y="530" text-anchor="middle">Evaluation-grid mean squared error (log scale)</text>',
                '<text x="24" y="552" font-size="12">Same training data, batch order and Adam steps; not an elapsed-time comparison.</text>', '</g></svg>'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mlp_comparison"))
    args = parser.parse_args()
    if (args.seed is None) == (not args.summarize):
        parser.error("Choose exactly one of --seed or --summarize")
    if args.summarize:
        summarize(args.output_dir)
    else:
        run_seed(args.seed, args.output_dir)
