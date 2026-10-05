"""Run every Python/Node behavior test: python test_kan.py [--report PATH]."""

import argparse
import json
from pathlib import Path
import sys
import unittest

import torch

from kan_layer import KANLayer
from kan_network import KAN
from kan_trainer import KANTrainer


class KANBehaviorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def test_kan_layer_creation(self):
        layer = KANLayer(in_features=5, out_features=3, grid_size=5)
        self.assertEqual((layer.in_features, layer.out_features, layer.grid_size), (5, 3, 5))
        self.assertEqual(tuple(layer.rbf_weight.shape), (3, 5, 6))
        torch.testing.assert_close(layer.grid, torch.linspace(-1, 1, 6))
        torch.testing.assert_close(layer.rbf_scaler, torch.ones(3, 5))
        self.assertAlmostEqual(layer.basis_sigma, 0.2)

    def test_kan_layer_forward(self):
        layer = KANLayer(in_features=3, out_features=2, grid_size=5)
        for batch_size in (10, 1):
            output = layer(torch.randn(batch_size, 3))
            self.assertEqual(tuple(output.shape), (batch_size, 2))
            self.assertTrue(torch.isfinite(output).all())

    def test_kan_network_creation(self):
        model = KAN(layers_hidden=[2, 5, 3, 1])
        self.assertEqual(len(model.layers), 3)
        self.assertEqual([(layer.in_features, layer.out_features) for layer in model.layers],
                         [(2, 5), (5, 3), (3, 1)])

    def test_kan_network_forward(self):
        model = KAN(layers_hidden=[2, 5, 1])
        self.assertEqual(tuple(model(torch.randn(8, 2)).shape), (8, 1))

    def test_kan_trainer_creation(self):
        model = KAN(layers_hidden=[1, 5, 1])
        trainer = KANTrainer(model, device=torch.device("cpu"), optimizer_name="Adam", lr=0.01)
        self.assertIs(trainer.model, model)
        self.assertEqual(trainer.optimizer_name, "Adam")

    def test_dataset_creation(self):
        trainer = KANTrainer(KAN([1, 5, 1]), device=torch.device("cpu"))
        x, y = trainer.create_dataset(func=lambda value: value ** 2, n_samples=100,
                                      input_dim=1, x_range=(-1, 1))
        self.assertEqual(tuple(x.shape), (100, 1))
        self.assertEqual(tuple(y.shape), (100, 1))
        self.assertTrue(torch.all(x >= -1) and torch.all(x <= 1))
        torch.testing.assert_close(y, x ** 2)

    def test_simple_training(self):
        trainer = KANTrainer(KAN([1, 3, 1], grid_size=3), device=torch.device("cpu"),
                             optimizer_name="Adam", lr=0.1)
        x, y = trainer.create_dataset(func=lambda value: 2 * value + 1, n_samples=50)
        history = trainer.train(x, y, epochs=5, batch_size=10, verbose=False)
        self.assertEqual(len(history["train_loss"]), 5)
        self.assertTrue(all(isinstance(loss, float) for loss in history["train_loss"]))
        self.assertTrue(all(torch.isfinite(torch.tensor(history["train_loss"]))))

    def test_regularization(self):
        layer = KANLayer(in_features=2, out_features=1, grid_size=3)
        reg_loss = layer.regularization_loss(regularize_activation=0.1, regularize_entropy=0.1)
        self.assertIsInstance(reg_loss, torch.Tensor)
        self.assertGreaterEqual(reg_loss.item(), 0)
        reg_loss.backward()
        self.assertIsNotNone(layer.rbf_weight.grad)

    def test_prediction(self):
        model = KAN([1, 3, 1])
        trainer = KANTrainer(model, device=torch.device("cpu"))
        x = torch.randn(5, 1)
        predictions = trainer.predict(x)
        self.assertEqual(tuple(predictions.shape), (5, 1))
        self.assertIsInstance(predictions, torch.Tensor)
        with torch.no_grad():
            torch.testing.assert_close(predictions, model(x))

    def test_different_activations(self):
        for activation in ("silu", "relu", "gelu", "tanh"):
            with self.subTest(activation=activation):
                layer = KANLayer(in_features=2, out_features=1, base_activation=activation)
                self.assertEqual(tuple(layer(torch.randn(3, 2)).shape), (3, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write numeric results and test status as JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    from tests.test_behavior import REPORT

    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test*.py", top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(REPORT)
    report.update(tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), success=result.wasSuccessful())
    print("Behavior report: " + json.dumps(report, sort_keys=True, allow_nan=False))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
