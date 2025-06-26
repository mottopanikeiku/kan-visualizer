# kolmogorov-arnold networks (kan) implementation

a pytorch implementation of kans, which replace boring linear layers with learnable functions on network edges.

## overview

traditional neural networks use linear transformations + fixed activations. kans put learnable functions on the edges instead, making them more interpretable and potentially way more efficient.

## features

- **core kan layer**: learnable functions using spline interpolation
- **flexible architecture**: multi-layer kans with configurable params
- **training framework**: complete training pipeline with reg
- **visualization**: built-in plotting for training history and results  
- **examples**: comprehensive examples for 1d/2d function approximation

## installation

```bash
pip install -r requirements.txt
```

## quick start

```python
import torch
from kan_network import KAN
from kan_trainer import KANTrainer

# create a kan model
model = KAN(
    layers_hidden=[2, 10, 1],  # input_dim=2, hidden=10, output_dim=1
    grid_size=5,               # number of grid points for splines
    spline_order=3             # spline order (3 for cubic)
)

# create trainer
trainer = KANTrainer(model, optimizer_name="Adam", lr=0.01)

# generate synthetic data
def target_func(x):
    return torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])

x_train, y_train = trainer.create_dataset(
    func=target_func, 
    n_samples=1000, 
    input_dim=2
)

# train the model
trainer.train(x_train, y_train, epochs=100, batch_size=64)

# make predictions
predictions = trainer.predict(x_test)
```

## architecture

### kan layer
- replaces linear layers with learnable spline functions
- each edge has its own little function parameterized by splines
- combines base activation + spline activation for flexibility

### kan network
- stacks multiple kan layers
- supports arbitrary depth and width
- built-in regularization for sparsity and smoothness

### training framework
- support for multiple optimizers (adam, adamw, lbfgs)
- automatic regularization with configurable weights
- built-in validation and early stopping
- training history tracking and visualization

## key parameters

- `grid_size`: number of grid points for spline interpolation (higher = more flexible)
- `spline_order`: spline order (typically 3 for cubic splines)
- `scale_base`: weight for base activation component
- `scale_spline`: weight for spline activation component
- `regularize_activation`: l1 regularization on spline weights
- `regularize_entropy`: entropy regularization for sparsity

## examples

run the examples to see kan in action:

```bash
python examples.py
```

this includes:
- 1d function approximation with visualization
- 2d function approximation

## theory

kans are based on the kolmogorov-arnold representation theorem, which says any multivariate continuous function can be represented as a composition of univariate functions. the key advantages:

1. **interpretability**: functions on edges can be visualized and analyzed
2. **efficiency**: can achieve good approximation with fewer parameters
3. **accuracy**: better approximation for smooth functions
4. **sparsity**: natural regularization leads to sparse networks

## file structure

```
kan-visualizer/
├── kan_layer.py      # core kan layer implementation
├── kan_network.py    # kan network and multi-layer variants
├── kan_trainer.py    # training framework and utilities
├── examples.py       # usage examples and demonstrations
├── requirements.txt  # python dependencies
└── README.md        # this file
```

## requirements

- python 3.8+
- pytorch 2.0+
- numpy
- matplotlib
- tqdm
- scipy

## license

see license file for details.

## references

- original kan paper: [kolmogorov-arnold networks](https://arxiv.org/abs/2404.19756)
- b-spline interpolation theory
- kolmogorov-arnold representation theorem 