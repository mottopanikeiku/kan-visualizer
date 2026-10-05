"""Gaussian edge functions inspired by KANs; this is not a B-spline layer."""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class KANLayer(nn.Module):
    """Sum a base activation and a learned Gaussian expansion on each edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        scale_base: float = 1.0,
        scale_rbf: float = 1.0,
        enable_standalone_scale_rbf: bool = True,
        base_activation: str = "silu",
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        if in_features < 1 or out_features < 1 or grid_size < 1:
            raise ValueError("Feature counts and grid_size must be positive")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must have increasing endpoints")
        activations = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh}
        if base_activation not in activations:
            raise ValueError(f"Unsupported activation: {base_activation}")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.scale_base = scale_base
        self.scale_rbf = scale_rbf
        self.enable_standalone_scale_rbf = enable_standalone_scale_rbf
        self.base_activation_name = base_activation
        self.base_activation = activations[base_activation]
        self.grid_range = grid_range
        self.register_buffer("grid", torch.linspace(*grid_range, grid_size + 1))
        self.basis_sigma = float((self.grid[1] - self.grid[0]) / 2)
        self.rbf_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + 1))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        if enable_standalone_scale_rbf:
            self.rbf_scaler = nn.Parameter(torch.ones(out_features, in_features))
        else:
            self.register_parameter("rbf_scaler", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.rbf_weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            self.base_weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            if self.rbf_scaler is not None:
                self.rbf_scaler.fill_(1.0)

    def gaussian_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate fixed Gaussian centers after clamping only the RBF input."""
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Expected (batch, {self.in_features}) inputs")
        clamped = x.clamp(*self.grid_range).unsqueeze(-1)
        return torch.exp(-((clamped - self.grid) ** 2) / (2 * self.basis_sigma ** 2))

    @property
    def effective_rbf_weight(self) -> torch.Tensor:
        if self.rbf_scaler is None:
            return self.rbf_weight
        return self.rbf_weight * self.rbf_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.gaussian_basis(x)
        base = F.linear(self.base_activation(x), self.base_weight)
        rbf = F.linear(basis.flatten(1), self.effective_rbf_weight.flatten(1))
        return self.scale_base * base + self.scale_rbf * rbf

    def edge_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Return signed contributions in (batch, output, input) order."""
        rbf = torch.einsum("big,oig->boi", self.gaussian_basis(x), self.effective_rbf_weight)
        base = self.base_activation(x).unsqueeze(1) * self.base_weight
        return self.scale_base * base + self.scale_rbf * rbf

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Coefficient L1 and coefficient-distribution entropy penalties."""
        loss = self.rbf_weight.new_zeros(())
        if regularize_activation > 0:
            loss = loss + regularize_activation * self.rbf_weight.abs().mean()
        if regularize_entropy > 0:
            p = self.rbf_weight.abs().softmax(dim=-1)
            loss = loss - regularize_entropy * (p * (p + 1e-8).log()).sum(dim=-1).mean()
        return loss
