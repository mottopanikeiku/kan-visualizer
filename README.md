# KAN visualizer

This is a small PyTorch Gaussian edge-network experiment with a browser viewer, inspired by [Liu et al.'s KAN paper](https://arxiv.org/abs/2404.19756), not its B-spline implementation.

**Question:** can the browser show the functions and activations that the trained Python network actually computes?

[kan_layer.py](kan_layer.py) learns Gaussian radial basis functions (RBFs) plus a base activation on each edge, with scaling applied before summing inputs. [export_for_web.py](export_for_web.py) exports the full model and sampled edge curves; [model-forward.js](web/js/model-forward.js) evaluates those parameters directly. Live inference shows real node activations, signed edge contributions, and the correct target for each task.

## Result

Python/Node parity passes at absolute and relative tolerance `1e-6`: maximum output error for the bundled float32 models is `6.48e-7`. A seeded toy `x²` task reduces MSE from `0.220325` to `0.000302613` in `160` CPU optimizer steps. All `14` tests pass; see [verification.json](results/verification.json). [Browser smoke checks](results/browser_smoke.json) passed across all models and views without runtime errors.

Regenerated synthetic demos give these MSEs on separate fixed evaluation grids, not on the training samples ([settings and results](results/demo_training.json)):

| Target | Initial MSE | Trained MSE |
| --- | ---: | ---: |
| `sin(3x) + 0.3 cos(10x)` | 0.563881 | 0.031081 |
| `sin(x) exp(-y²)` | 0.202277 | 0.002112 |
| `sin(xy) + 0.5 tanh(x-y)` | 0.599373 | 0.019927 |

These results establish numerical agreement and small-task learning, not superiority over an MLP.

## Reproduce

Requires Python, `uv`, Node, and a browser. CPU only; no GPU or paid service. The local training run took about `12` seconds and peaked at `355` MiB resident memory; tests took about `3` seconds ([runtime measurements](results/runtime.json)). The browser loads plotting libraries from public CDNs, so viewing needs internet access.

```bash
uv venv && uv pip install --python .venv/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv/bin/python -r requirements.txt
nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python export_for_web.py && nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python test_kan.py --report results/verification.json
nice -n 19 .venv/bin/python -m http.server 8002 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8002`. The viewer also works with the committed exports without retraining. Test failures return a nonzero exit status.

## Limitations

- Fixed Gaussian centers, not B-splines; no grid adaptation, spline fitting, pruning, or symbolic extraction.
- The RBF branch clamps inputs to its grid range; the base activation does not. Boundary behavior can limit fits outside that range.
- Each demo is one seeded run on one synthetic task. There is no real-data or parameter-matched MLP comparison.
- Graph edge RMS is a sampled function summary, not feature importance; its connectivity animation is only an illustration.
- Old spline-named Python arguments and incomplete JSON exports are intentionally unsupported. Models must be regenerated with the current exporter.

## Prior work and details

The learned-univariate-edge design builds on [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). Gaussian parameterization differs from the paper's B-spline model. [FastKAN](https://github.com/ZiyaoLi/fast-kan) explores Gaussian RBF replacements; this repository does not claim to reproduce its implementation or results.

[Implementation, export equations, display meanings, and the next comparison](docs/GUIDE.md). Code is [MIT licensed](LICENSE).
