# Gaussian edge-network implementation

This is a small KAN-inspired experiment, not a reproduction of the B-spline architecture in [Liu et al., *KAN: Kolmogorov-Arnold Networks*](https://arxiv.org/abs/2404.19756). The code uses fixed Gaussian radial basis functions (RBFs). No efficiency, interpretability, or accuracy advantage over an MLP has been measured here.

## Layer equation

For input coordinate `x_i` and output `j`, the signed edge contribution is

```text
edge[j,i](x_i) = scale_base * base_weight[j,i] * activation(x_i)
              + scale_rbf * rbf_scaler[j,i]
                * sum_k rbf_weight[j,i,k]
                        * exp(-(clamp(x_i, lo, hi)-grid[k])² / (2*sigma²))
output[j] = sum_i edge[j,i](x_i)
```

`grid_size` is the number of intervals, so the fixed uniform grid has `grid_size + 1` centers. `sigma` is half the grid spacing. Only the RBF input is clamped; the base activation sees the original input. The available base activations are SiLU, ReLU, GELU, and tanh. Disabling the standalone scaler sets its effective value to one. Both global scale factors and every per-edge scaler are exported.

The implementation is in `../kan_layer.py`. `forward` uses matrix operations without retaining intermediate edge tensors. `edge_contributions` returns signed per-edge values for display and checks. `../kan_network.py` stacks layers. There is no adaptive grid, spline fitting, pruning, or symbolic extraction API. The coefficient L1 and coefficient-distribution entropy penalties are not a smoothness guarantee.

## Training and export

```python
import torch
from kan_network import KAN
from kan_trainer import KANTrainer
from export_for_web import export_kan_for_visualization

model = KAN([2, 15, 1], grid_size=5)
trainer = KANTrainer(model, device=torch.device("cpu"), optimizer_name="Adam", lr=0.01)
x, y = trainer.create_dataset(
    lambda x: torch.sin(x[:, :1]) * torch.exp(-x[:, 1:2] ** 2),
    n_samples=1500, input_dim=2, x_range=(-2, 2),
)
trainer.train(x, y, epochs=150, batch_size=128, verbose=False)
export_kan_for_visualization(model, trainer, target_id="2d_gaussian")
```

`../export_for_web.py` regenerates all bundled models with fixed CPU seeds and writes separate-grid MSE to `../results/demo_training.json`. The training histories contain average batch losses, not the independent evaluation-grid MSE. A vector-valued scalar target is converted to a column before training to avoid accidental loss broadcasting.

Exports declare `basis: gaussian_rbf`. The browser evaluates coefficients, base activation, input clamp, and scaling directly in `../web/js/model-forward.js`; it does not interpolate display curves. The same JavaScript module runs under Node for parity tests. Unsupported model schemas or activations raise errors. GELU uses an erf approximation in JavaScript; parity allows absolute and relative error of `1e-6` against PyTorch float32.

## Display meanings

- Live inference nodes show actual current activations, and edge highlighting follows absolute signed contributions for the current input.
- The activation-flow plot reports mean absolute activation per layer, including the input layer.
- Edge-function curves include both base and RBF terms. They are sampled functions, not fitted splines.
- A graph summary over sampled edge values is not causal feature importance.
- Targets are selected by exported task identity, not merely by input dimension. The two-dimensional wave is `sin(x) * exp(-y*y)`; the interaction task is `sin(x*y) + 0.5*tanh(x-y)`.

## Verification and next comparison

Run `nice -n 19 .venv/bin/python test_kan.py --report results/verification.json` from the repository root with Node installed. The runner returns a failure status if a check fails. It compares fixed Python/Node outputs, internal activations, and edge contributions, checks exported curves and targets, and requires a seeded toy training task to reduce MSE by at least a factor of ten. This is a correctness and learnability check, not a generalization benchmark.

The next useful experiment is a parameter-matched MLP comparison on these same functions and evaluation grids, including multiple seeds and elapsed CPU time. No such comparison has been run. It needs only local CPU and no paid service.
