"""Check baseline sizes, evaluation grids, and paired result invariants."""

import json
from pathlib import Path
import unittest

import torch
from torch import nn

from compare_mlp import CONFIGS, FAMILIES, SEEDS, evaluation_grid, make_model
from export_for_web import TASKS

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def test_parameter_matching_and_depth(self):
        for task, config in CONFIGS.items():
            counts = {}
            for family in FAMILIES:
                model = make_model(config, family)
                counts[family] = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(tuple(model(torch.zeros(3, config["kan"][0])).shape), (3, 1))
                if family != "gaussian_edge":
                    self.assertIsInstance(model[-1], nn.Linear)
                    self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model),
                                     len(config["kan"]) - 1)
            for family in FAMILIES[1:]:
                self.assertLess(abs(counts[family] / counts["gaussian_edge"] - 1), 0.003)
            self.assertEqual(counts["mlp_tanh"], counts["mlp_silu"])

    def test_fixed_evaluation_grid_and_scalar_targets(self):
        for task, config in CONFIGS.items():
            dim = config["kan"][0]
            grid = evaluation_grid(dim)
            self.assertEqual(tuple(grid.shape), (201, 1) if dim == 1 else (961, 2))
            self.assertEqual(grid.min().item(), -2)
            self.assertEqual(grid.max().item(), 2)
            torch.testing.assert_close(grid, evaluation_grid(dim))
            self.assertEqual(tuple(TASKS[task][3](grid).shape), (len(grid), 1))

    def test_committed_results_are_complete_and_paired(self):
        directory = ROOT / "results/mlp_comparison"
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["seeds"], SEEDS)
        all_runs = []
        for seed in SEEDS:
            report = json.loads((directory / f"seed_{seed}.json").read_text())
            self.assertEqual(report["seed"], seed)
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["threads"], 1)
            runs = report["runs"]
            self.assertEqual(len(runs), len(CONFIGS) * len(FAMILIES))
            for task, config in CONFIGS.items():
                paired = [run for run in runs if run["task_id"] == task]
                self.assertEqual({run["family"] for run in paired}, set(FAMILIES))
                for field in ["training_inputs_sha256", "evaluation_inputs_sha256", "minibatch_order_sha256"]:
                    self.assertEqual(len({run[field] for run in paired}), 1)
                steps = config["epochs"] * ((config["samples"] + config["batch_size"] - 1) // config["batch_size"])
                for run in paired:
                    self.assertEqual(run["optimizer_steps"], steps)
                    self.assertEqual(run["seed"], seed)
                    self.assertGreaterEqual(run["final_grid"]["mse"], 0)
                    self.assertTrue(torch.isfinite(torch.tensor(list(run["final_grid"].values()))).all())
            all_runs.extend(runs)
        self.assertEqual(summary["runs"], len(all_runs))
        import statistics
        for row in summary["rows"]:
            group = [run for run in all_runs if run["task_id"] == row["task_id"] and run["family"] == row["family"]]
            values = [run["final_grid"]["mse"] for run in group]
            self.assertEqual(row["median_grid_mse"], statistics.median(values))
            self.assertEqual(row["min_grid_mse"], min(values))
            self.assertEqual(row["max_grid_mse"], max(values))
            self.assertEqual({run["parameters"] for run in group}, {row["parameters"]})
            self.assertEqual({run["optimizer_steps"] for run in group}, {row["optimizer_steps"]})
