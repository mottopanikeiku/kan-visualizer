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

## Browser deployment

The static viewer needs no build or model training. `.github/workflows/pages.yml` checks JavaScript syntax and exported JSON on pull requests, then uploads the entire `web/` directory, including `web/data/`. Pushes to `main` and manual workflow runs deploy that artifact with GitHub's Pages actions. Repository Pages settings must already allow GitHub Actions deployment; the workflow does not change them.

Styles, scripts, and model fetches use document-relative paths, so the same files work at `/kan-visualizer/` without a root-path rewrite. To preview that prefix locally from the repository root:

```bash
preview=$(mktemp -d)
ln -s "$PWD/web" "$preview/kan-visualizer"
nice -n 19 python -m http.server 8002 --bind 127.0.0.1 --directory "$preview"
```

Open `http://127.0.0.1:8002/kan-visualizer/`. The page uses system font fallbacks and exact D3 `7.9.0` and Plotly `2.35.2` CDN URLs with SHA-384 integrity checks. Internet access is needed for those two libraries, but inference uses the checked-in model parameters entirely in the browser. Select a model, choose **live inference**, and change its input sliders to inspect the actual output and task target. The deployment smoke screenshot is in `docs/assets/pages-inference.png`.

`training_history.train_loss` is the exporter's recorded batch-average loss series; the training tab plots it directly. An empty `val_loss` array is not presented as a validation measurement. The repeatable prefix smoke check exercises all three models and four views, compares the training trace with the exported series, changes inference sliders with keyboard events, and records errors, model outputs, and the screenshot:

```bash
uv run --no-project --with playwright playwright install chromium
uv run --no-project --with playwright python tests/pages_smoke.py
```

If Chromium is already installed elsewhere, set `KAN_CHROMIUM_PATH` to its executable for the second command. The smoke check writes `results/browser_pages.json` and `docs/assets/pages-inference.png`.


## Verification and next comparison

Run `nice -n 19 .venv/bin/python test_kan.py --report results/verification.json` from the repository root with Node installed. The runner returns a failure status if a check fails. It compares fixed Python/Node outputs, internal activations, and edge contributions, checks exported curves and targets, and requires a seeded toy training task to reduce MSE by at least a factor of ten. This is a correctness and learnability check, not a generalization benchmark.

The next useful experiment is a parameter-matched MLP comparison on these same functions and evaluation grids, including multiple seeds and elapsed CPU time. No such comparison has been run. It needs only local CPU and no paid service.
