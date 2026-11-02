# Gaussian edge-network implementation

The model uses fixed Gaussian radial basis functions (RBFs) on its edges. [Liu et al.'s KAN](https://arxiv.org/abs/2404.19756) uses B-splines. The [MLP comparison](../results/mlp_comparison/summary.json) measures regression error on the three demo tasks.

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

[kan_layer.py](../kan_layer.py) implements the layer. `forward` uses matrix operations without retaining intermediate edge tensors; `edge_contributions` returns signed per-edge values for display. [kan_network.py](../kan_network.py) stacks layers.

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

[export_for_web.py](../export_for_web.py) regenerates the bundled models with fixed CPU seeds and writes evaluation-grid MSE to [results/demo_training.json](../results/demo_training.json). Training histories contain average batch losses. Scalar targets are converted to columns to avoid loss broadcasting.

Exports declare `basis: gaussian_rbf`. [model-forward.js](../web/js/model-forward.js) evaluates the coefficients, base activation, clamp and scaling directly. The same module runs under Node for parity tests. Unsupported schemas and activations raise errors. GELU uses an erf approximation; parity tests allow absolute and relative error of `1e-6` against PyTorch float32.

## Display meanings

- Live inference nodes show activations at the current input. Edges show signed contributions, with width proportional to magnitude.
- The activation-flow plot reports mean absolute activation per layer, including the input layer.
- Edge curves include the base and RBF terms. Network-graph widths summarize RMS across those samples.
- Targets follow the exported task identity. The two-dimensional wave is `sin(x) * exp(-y*y)`; the interaction task is `sin(x*y) + 0.5*tanh(x-y)`.

The node and edge menus mirror graph clicks. Expand **Read numerical activations and edge contributions** for labelled tables that update with the sliders. Reduced-motion mode disables sweeps and particles while keeping manual controls.

## Browser deployment

The static viewer needs no build or model training. [The Pages workflow](../.github/workflows/pages.yml) checks Python/Node behavior, comparison data, JavaScript syntax and desktop/mobile browser interactions before deploying `web/`.

From the repository root:

```sh
python3 -m http.server 8002 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8002/`. Scripts and model paths are relative, so deployment under `/kan-visualizer/` works too. D3 `7.9.0` and Plotly `2.35.2` load from CDN URLs with SHA-384 integrity checks; inference runs in the browser.

The browser suite covers all three models and four views at desktop and mobile widths. It checks keyboard selections, numerical tables, training traces, reduced motion and model-load ordering:

```bash
uv run --no-project --with playwright==1.63.0 playwright install chromium
uv run --no-project --with playwright==1.63.0 python tests/pages_smoke.py
```

If Chromium is already installed elsewhere, set `KAN_CHROMIUM_PATH` to its executable for the second command. The smoke check writes a report and screenshot to `_smoke/`.

## Tests and MLP comparison

Run `.venv/bin/python test_kan.py` from the repository root with Node installed. The suite compares Python and JavaScript outputs, activations and edge contributions. It also checks exported curves, optimizer behavior, regularization and checkpoint restoration, and requires a seeded training task to reduce MSE tenfold.

[compare_mlp.py](../compare_mlp.py) uses the demo architectures and training settings. Dense MLPs have the same hidden-layer depth, linear output layers and Tanh or SiLU activations. Widths are chosen from parameter counts, including biases and edge scalers; counts differ by less than 0.3%.

For each of five seeds, all three model families receive identical uniform inputs on `[-2, 2]`, targets and minibatch orders. Initialization uses a separate generator. Adam uses the same task-specific learning rate and step budget for every model. The final iterate is evaluated, without checkpoint selection or tuning on the evaluation grid.

Evaluation uses 201 evenly spaced inputs in one dimension and a 31-by-31 grid in two dimensions. Per-seed files contain MSE, MAE, maximum absolute error, parameter counts, steps and hashes of the inputs and minibatch orders. The summary checks that runs are complete and paired before calculating medians and ranges.

```bash
for seed in 1729 1730 1731 1732 1733; do
  nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compare_mlp.py --seed "$seed"
done
.venv/bin/python compare_mlp.py --summarize
```

The tasks are small noiseless functions with one optimizer setting each. The Gaussian centers are fixed, and the RBF branch clamps to `[-1, 1]` while inputs span `[-2, 2]`. The browser keeps its original single-seed exports. Edge width in the graph shows sampled RMS, which is not feature importance.
