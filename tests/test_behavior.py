"""Analytic Gaussian-edge, Python/Node parity, and CPU learning regressions.

Every parity comparison uses atol=1e-6 and rtol=1e-6. The independent
reference clamps only RBF inputs and scales each edge before summation.
"""

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
import torch.nn.functional as F

from export_for_web import create_sample_datasets_for_web, export_kan_for_visualization
from kan_layer import KANLayer
from kan_network import KAN
from kan_trainer import KANTrainer


ATOL = 1e-6
RTOL = 1e-6
SEED = 1729
REPORT = {"seed": SEED, "atol": ATOL, "rtol": RTOL,
          "parity_max_absolute_error": None, "learning_start_mse": None,
          "learning_end_mse": None}
ROOT = Path(__file__).resolve().parents[1]
INPUTS = [[-3.25, 2.75], [-1.0, 1.0], [-0.73, 0.137], [0.0, 0.0],
          [0.51, -0.29], [1.0, -1.0], [2.4, -2.1]]
ACTIVATIONS = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh}


def reference_edges(layer, inputs, activation):
    """Derive full edges independently of layer's evaluator/forward methods."""
    sigma = (layer.grid[1] - layer.grid[0]) / 2
    clamped = inputs.clamp(*layer.grid_range).unsqueeze(-1)
    basis = torch.exp(-((clamped - layer.grid) ** 2) / (2 * sigma ** 2))
    rbf = (basis.unsqueeze(1) * layer.rbf_weight.unsqueeze(0)).sum(dim=-1)
    if layer.rbf_scaler is not None:
        rbf = rbf * layer.rbf_scaler.unsqueeze(0)
    base = ACTIVATIONS[activation](inputs).unsqueeze(1) * layer.base_weight.unsqueeze(0)
    return layer.scale_base * base + layer.scale_rbf * rbf


def fixed_model(activation, disable_scalers):
    model = KAN([2, 3, 2], grid_size=4, base_activation=activation).double()
    for index, old_layer in enumerate(model.layers):
        layer = KANLayer(old_layer.in_features, old_layer.out_features, grid_size=4,
                         grid_range=(-1.5, 1.5) if index else (-1, 1),
                         scale_base=1.7 if index == 0 else -0.6,
                         scale_rbf=0.45 if index == 0 else 1.3,
                         enable_standalone_scale_rbf=not disable_scalers,
                         base_activation=activation).double()
        with torch.no_grad():
            base_indices = torch.arange(layer.base_weight.numel(), dtype=torch.float64)
            layer.base_weight.copy_((0.9 * torch.cos(base_indices + 0.4 + index))
                                    .reshape_as(layer.base_weight))
            indices = torch.arange(layer.rbf_weight.numel(), dtype=torch.float64)
            layer.rbf_weight.copy_((0.65 * torch.sin(indices * 0.71 + 0.3 + index))
                                   .reshape_as(layer.rbf_weight))
            if layer.rbf_scaler is not None:
                # Different magnitudes and signs on each input edge expose scaling-after-sum bugs.
                scalers = torch.tensor([0.2, -1.4, 2.1, 0.65, -0.35, 1.7], dtype=torch.float64)
                layer.rbf_scaler.copy_(scalers.reshape_as(layer.rbf_scaler))
        model.layers[index] = layer
    return model


class GaussianParityTests(unittest.TestCase):
    def test_clean_cutover_api(self):
        for removed in ("spline_order", "scale_noise", "grid_eps", "scale_spline"):
            with self.subTest(parameter=removed):
                with self.assertRaises(TypeError):
                    KANLayer(2, 1, **{removed: 1})
                with self.assertRaises(TypeError):
                    KAN([2, 1], **{removed: 1})
        layer = KANLayer(2, 1, enable_standalone_scale_rbf=False)
        self.assertIsNone(layer.rbf_scaler)
        self.assertNotIn("rbf_scaler", dict(layer.named_parameters()))

    def test_python_node_outputs_activations_edges_and_exported_curves(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node is required for Python/browser parity; do not skip this test")
        cases = []
        expected = []
        with tempfile.TemporaryDirectory() as directory:
            for activation in ACTIVATIONS:
                for disable_scalers in (False, True):
                    name = f"{activation}-{'disabled' if disable_scalers else 'per-edge'}-scalers"
                    model = fixed_model(activation, disable_scalers)
                    with torch.no_grad():
                        inputs = torch.tensor(INPUTS, dtype=torch.float64)
                        values = inputs
                        activations = [inputs.clone()]
                        edges = []
                        for layer in model.layers:
                            independent = reference_edges(layer, values, activation)
                            torch.testing.assert_close(layer.edge_contributions(values), independent,
                                                       atol=ATOL, rtol=RTOL)
                            output = layer(values)
                            torch.testing.assert_close(output, independent.sum(dim=-1), atol=ATOL, rtol=RTOL)
                            edges.append(independent)
                            activations.append(output)
                            values = output
                        torch.testing.assert_close(model(inputs), values, atol=ATOL, rtol=RTOL)
                    export_path = Path(directory) / f"{name}.json"
                    exported = export_kan_for_visualization(model, save_path=str(export_path),
                                                             target_id="2d_complex")
                    self.assertEqual(exported, json.loads(export_path.read_text()))
                    metadata = exported["metadata"]
                    self.assertEqual(metadata["basis"], "gaussian_rbf")
                    self.assertEqual(metadata["architecture"], [2, 3, 2])
                    self.assertEqual(metadata["grid_size"], 4)
                    self.assertEqual(metadata["num_layers"], 2)
                    self.assertEqual(metadata["target_id"], "2d_complex")
                    self.assertEqual(metadata["total_parameters"], sum(p.numel() for p in model.parameters()))
                    self.assertNotIn("spline_order", metadata)
                    for layer_index, (layer, layer_data) in enumerate(zip(model.layers, exported["layers"])):
                        self.assertEqual(layer_data["layer_index"], layer_index)
                        self.assertEqual(layer_data["base_activation"], activation)
                        self.assertEqual(layer_data["basis_sigma"], layer.basis_sigma)
                        self.assertEqual(layer_data["scale_base"], layer.scale_base)
                        self.assertEqual(layer_data["scale_rbf"], layer.scale_rbf)
                        self.assertEqual(layer_data["grid_range"], list(layer.grid_range))
                        self.assertEqual(layer_data["grid_points"], layer.grid.tolist())
                        self.assertEqual(layer_data["rbf_coefficients"], layer.rbf_weight.tolist())
                        self.assertEqual(layer_data["base_weights"], layer.base_weight.tolist())
                        self.assertEqual(layer_data["rbf_scalers"], None if disable_scalers else layer.rbf_scaler.tolist())
                        self.assertNotIn("spline_evaluations", layer_data)
                        indices = set()
                        for curve in layer_data["edge_evaluations"]:
                            in_index, out_index = curve["input_idx"], curve["output_idx"]
                            self.assertIn(in_index, range(layer.in_features))
                            self.assertIn(out_index, range(layer.out_features))
                            self.assertNotIn((out_index, in_index), indices)
                            indices.add((out_index, in_index))
                            curve_inputs = torch.zeros(len(curve["x_values"]), layer.in_features,
                                                       dtype=torch.float64)
                            curve_inputs[:, in_index] = torch.tensor(curve["x_values"], dtype=torch.float64)
                            with torch.no_grad():
                                full_edge = reference_edges(layer, curve_inputs, activation)[:, out_index, in_index]
                            torch.testing.assert_close(torch.tensor(curve["y_values"], dtype=torch.float64),
                                                       full_edge, atol=ATOL, rtol=RTOL)
                        self.assertEqual(len(indices), layer.in_features * layer.out_features)
                    cases.append({"model": exported, "inputs": INPUTS, "name": name})
                    expected.append({"output": values, "activations": activations, "edges": edges})

            datasets_path = Path(directory) / "datasets.json"
            datasets = create_sample_datasets_for_web(save_path=str(datasets_path))
            self.assertEqual(datasets, json.loads(datasets_path.read_text()))
            self.assertEqual(set(datasets), {"1d_sine_wave", "2d_gaussian", "2d_complex"})
            for target_id, dataset in datasets.items():
                self.assertEqual(dataset["target_id"], target_id)
                self.assertEqual(dataset["input_dim"], 1 if target_id == "1d_sine_wave" else 2)
                self.assertTrue(dataset["samples"])
                for sample in dataset["samples"]:
                    coordinates = sample["input"]
                    self.assertEqual(len(coordinates), dataset["input_dim"])
                    if target_id == "1d_sine_wave":
                        target = math.sin(3 * coordinates[0]) + 0.3 * math.cos(10 * coordinates[0])
                    elif target_id == "2d_gaussian":
                        target = math.sin(coordinates[0]) * math.exp(-coordinates[1] ** 2)
                    else:
                        target = math.sin(coordinates[0] * coordinates[1]) + 0.5 * math.tanh(coordinates[0] - coordinates[1])
                    self.assertLessEqual(abs(sample["output"] - target), ATOL + RTOL * abs(target))

            process = subprocess.run([node, str(ROOT / "tests" / "node_evaluator_checks.js")],
                                     input=json.dumps({"cases": cases, "datasets": datasets}), text=True,
                                     capture_output=True, cwd=ROOT, timeout=60, check=False)
            self.assertEqual(process.returncode, 0,
                             f"Node evaluator assertions failed:\n{process.stdout}\n{process.stderr}")
            actual = json.loads(process.stdout)

        self.assertEqual(len(actual["results"]), len(cases))
        max_error = 0.0
        for case, python_values, node_values in zip(cases, expected, actual["results"]):
            self.assertEqual(len(node_values), len(INPUTS))
            for input_index, result in enumerate(node_values):
                with self.subTest(model=case["name"], input=INPUTS[input_index]):
                    self.assertEqual(len(result["activations"]), len(python_values["activations"]))
                    self.assertEqual(len(result["edges"]), len(python_values["edges"]))
                    pairs = [(result["output"], python_values["output"][input_index])]
                    pairs.extend((value, reference[input_index]) for value, reference in
                                 zip(result["activations"], python_values["activations"]))
                    pairs.extend((value, reference[input_index]) for value, reference in
                                 zip(result["edges"], python_values["edges"]))
                    for value, reference in pairs:
                        tensor = torch.tensor(value, dtype=torch.float64)
                        self.assertEqual(tensor.shape, reference.shape)
                        self.assertTrue(torch.isfinite(tensor).all())
                        max_error = max(max_error, (tensor - reference).abs().max().item())
                        torch.testing.assert_close(tensor, reference, atol=ATOL, rtol=RTOL)
        REPORT.update(parity_max_absolute_error=max_error, parity_models=len(cases),
                      parity_inputs_per_model=len(INPUTS),
                      exported_curve_max_absolute_error=actual["curveMaxAbsoluteError"],
                      exported_curve_points_checked=actual["curvesChecked"])

    def test_bundled_trained_float32_models(self):
        """Check the exact committed demo weights, not only constructed fixtures."""
        cases = []
        expected = []
        for name in ("model_1d", "model_2d", "model_complex"):
            exported = json.loads((ROOT / "web" / "data" / f"{name}.json").read_text())
            model = KAN(exported["metadata"]["architecture"],
                        grid_size=exported["metadata"]["grid_size"])
            for layer, data in zip(model.layers, exported["layers"]):
                with torch.no_grad():
                    layer.base_weight.copy_(torch.tensor(data["base_weights"]))
                    layer.rbf_weight.copy_(torch.tensor(data["rbf_coefficients"]))
                    layer.rbf_scaler.copy_(torch.tensor(data["rbf_scalers"]))
            inputs = [[row[0]] for row in INPUTS] if model.layers_hidden[0] == 1 else INPUTS
            with torch.no_grad():
                expected.append(model(torch.tensor(inputs)).double())
            cases.append({"model": exported, "inputs": inputs, "name": name})
        datasets = json.loads((ROOT / "web" / "data" / "datasets.json").read_text())
        process = subprocess.run(["node", str(ROOT / "tests" / "node_evaluator_checks.js")],
                                 input=json.dumps({"cases": cases, "datasets": datasets}),
                                 text=True, capture_output=True, cwd=ROOT, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        actual = json.loads(process.stdout)
        max_error = 0.0
        for outputs, reference in zip(actual["results"], expected):
            values = torch.tensor([result["output"] for result in outputs], dtype=torch.float64)
            torch.testing.assert_close(values, reference, atol=ATOL, rtol=RTOL)
            max_error = max(max_error, (values - reference).abs().max().item())
        REPORT.update(bundled_float32_models=len(cases),
                      bundled_float32_max_absolute_error=max_error)


class CPULearningTests(unittest.TestCase):
    def test_optimizer_reduces_fixed_grid_mse_tenfold(self):
        torch.manual_seed(SEED)
        model = KAN([1, 1], grid_size=8, scale_base=0.0)
        # Start at a known zero prediction, not at a lucky initialized fit.
        with torch.no_grad():
            model.layers[0].base_weight.zero_()
            model.layers[0].rbf_weight.zero_()
        trainer = KANTrainer(model, device=torch.device("cpu"), optimizer_name="Adam", lr=0.04)
        inputs = torch.linspace(-1, 1, 41).unsqueeze(1)
        targets = inputs.square()
        initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        start_mse = trainer.evaluate(inputs, targets)
        steps = 160
        history = trainer.train(inputs, targets, epochs=steps, batch_size=len(inputs), verbose=False)
        end_mse = trainer.evaluate(inputs, targets)
        self.assertTrue(math.isfinite(start_mse) and math.isfinite(end_mse))
        REPORT.update(learning_start_mse=start_mse, learning_end_mse=end_mse,
                      learning_steps=steps, learning_samples=len(inputs), learning_target="x^2",
                      learning_architecture=[1, 1], learning_device="cpu",
                      learning_loss_reduction_factor=start_mse / max(end_mse, 1e-30))
        self.assertEqual(len(history["train_loss"]), steps)
        self.assertTrue(all(math.isfinite(loss) for loss in history["train_loss"]))
        self.assertTrue(any(not torch.equal(value.detach(), initial_parameters[name])
                            for name, value in model.named_parameters()))
        self.assertGreater(start_mse, 0.1)
        self.assertLessEqual(end_mse, start_mse / 10,
                             f"CPU optimizer failed to learn: start MSE={start_mse}, end MSE={end_mse}")
