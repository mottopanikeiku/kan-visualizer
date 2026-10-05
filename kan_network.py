"""Networks of learned Gaussian edge functions, inspired by the KAN paper."""

from typing import List, Tuple

import torch
from torch import nn

from kan_layer import KANLayer


class KAN(nn.Module):
    """Stack fixed-grid Gaussian layers with a common configuration."""

    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        scale_base: float = 1.0,
        scale_rbf: float = 1.0,
        base_activation: str = "silu",
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("Specify at least an input and an output layer")
        self.layers_hidden = list(layers_hidden)
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.layers = nn.ModuleList([
            KANLayer(
                in_features=inputs,
                out_features=outputs,
                grid_size=grid_size,
                scale_base=scale_base,
                scale_rbf=scale_rbf,
                base_activation=base_activation,
                grid_range=grid_range,
            )
            for inputs, outputs in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy)
                   for layer in self.layers)


class MultilayerKAN(nn.Module):
    """A Gaussian network with separately configured layers."""

    def __init__(self, layer_configs: List[dict]):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(**config) for config in layer_configs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy)
                   for layer in self.layers)
