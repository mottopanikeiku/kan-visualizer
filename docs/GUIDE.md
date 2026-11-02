# Gaussian edge-network implementation

The model puts fixed Gaussian radial basis functions (RBFs) on its edges where [Liu et al.'s KAN](https://arxiv.org/abs/2404.19756) uses B-splines, and the [MLP comparison](../results/mlp_comparison/summary.json) measures its regression error on the three demo tasks.

## Layer equation

For input coordinate `x_i` and output `j`, the signed edge contribution is

```text
edge[j,i](x_i) = scale_base * base_weight[j,i] * activation(x_i)
              + scale_rbf * rbf_scaler[j,i]
                * sum_k rbf_weight[j,i,k]
                        * exp(-(clamp(x_i, lo, hi)-grid[k])² / (2*sigma²))
output[j] = sum_i edge[j,i](x_i)
```

`grid_size` is the number of intervals, so the fixed uniform grid has `grid_size + 1` centers, and `sigma` is half the grid spacing. Only the RBF input is clamped, while the base activation, which can be SiLU, ReLU, GELU or tanh, sees the original input. Disabling the standalone scaler sets its effective value to one, and both global scale factors and every per-edge scaler are exported.

[kan_layer.py](../kan_layer.py) implements the layer and [kan_network.py](../kan_network.py) stacks layers into a network. Within the layer, `forward` uses matrix operations without keeping the intermediate edge tensors, while `edge_contributions` returns the signed per-edge values that the viewer displays.

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

[export_for_web.py](../export_for_web.py) regenerates the bundled models with fixed CPU seeds and writes their evaluation-grid MSE to [results/demo_training.json](../results/demo_training.json). The training histories it records contain average batch losses, and scalar targets are converted to columns so that the loss doesn't broadcast.

Exports declare `basis: gaussian_rbf`, and [model-forward.js](../web/js/model-forward.js) evaluates the coefficients, base activation, clamp and scaling directly, raising an error for any schema or activation it doesn't support. Its GELU uses an erf approximation, and the same module runs under Node for the parity tests, which allow absolute and relative error of `1e-6` against PyTorch float32.

## Display meanings

- Live inference nodes show activations at the current input, and edges show signed contributions with width proportional to magnitude.
- The activation-flow plot reports mean absolute activation per layer, including the input layer.
- Edge curves include both the base and RBF terms, and network-graph widths summarize the RMS across those samples.
- Targets follow the exported task identity, so the two-dimensional wave is `sin(x) * exp(-y*y)` and the interaction task is `sin(x*y) + 0.5*tanh(x-y)`.

The node and edge menus mirror clicks on the graph, and expanding **Read numerical activations and edge contributions** shows labelled tables that update with the sliders. Reduced-motion mode disables sweeps and particles while keeping the manual controls.

## Browser deployment

The static viewer needs no build step or model training, and [the Pages workflow](../.github/workflows/pages.yml) checks Python/Node behavior, comparison data, JavaScript syntax and desktop/mobile browser interactions before it deploys `web/`. To serve it locally, run this from the repository root and open `http://127.0.0.1:8002/`:

```sh
python3 -m http.server 8002 --bind 127.0.0.1 --directory web
```

Because scripts and model paths are relative, the same files also work when deployed under `/kan-visualizer/`. D3 `7.9.0` and Plotly `2.35.2` load from CDN URLs with SHA-384 integrity checks, and all inference runs in the browser.

The browser suite covers all three models and four views at desktop and mobile widths, checking keyboard selections, numerical tables, training traces, reduced motion and model-load ordering:

```bash
uv run --no-project --with playwright==1.63.0 playwright install chromium
uv run --no-project --with playwright==1.63.0 python tests/pages_smoke.py
```

If Chromium is already installed elsewhere, set `KAN_CHROMIUM_PATH` to its executable for the second command. The smoke check writes a report and screenshot to `_smoke/`.

## Tests and MLP comparison

Run `.venv/bin/python test_kan.py` from the repository root with Node installed. The suite compares Python and JavaScript outputs, activations and edge contributions. It also checks exported curves, optimizer behavior, regularization and checkpoint restoration, and requires a seeded training task to reduce MSE tenfold.

[compare_mlp.py](../compare_mlp.py) uses the demo architectures and training settings, and its dense MLPs have the same hidden-layer depth, linear output layers and Tanh or SiLU activations. Their widths are chosen so that parameter counts, including biases and edge scalers, differ by less than 0.3%.

For each of five seeds, all three model families receive identical uniform inputs on `[-2, 2]`, targets and minibatch orders, while initialization draws from a separate generator. Adam uses the same task-specific learning rate and step budget for every model, and the final iterate is evaluated without checkpoint selection or tuning on the evaluation grid.

Evaluation uses 201 evenly spaced inputs in one dimension and a 31-by-31 grid in two dimensions. Per-seed files contain MSE, MAE, maximum absolute error, parameter counts, steps and hashes of the inputs and minibatch orders. The summary checks that runs are complete and paired before calculating medians and ranges.

```bash
for seed in 1729 1730 1731 1732 1733; do
  nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compare_mlp.py --seed "$seed"
done
.venv/bin/python compare_mlp.py --summarize
```

The tasks are small noiseless functions with one optimizer setting each. The Gaussian centers are fixed, and the RBF branch clamps to `[-1, 1]` while inputs span `[-2, 2]`. The browser still shows its original single-seed exports, and edge width in the graph shows sampled RMS, which is not a measure of feature importance.
