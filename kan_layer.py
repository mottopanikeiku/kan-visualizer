import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class KANLayer(nn.Module):
    """
    kolmogorov-arnold network layer
    
    replaces boring linear layers with learnable functions on each edge.
    each connection has its own little function that learns via splines.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: str = "silu",
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        super(KANLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        
        # setup base activation
        if base_activation == "silu":
            self.base_activation = F.silu
        elif base_activation == "relu":
            self.base_activation = F.relu
        elif base_activation == "gelu":
            self.base_activation = F.gelu
        elif base_activation == "tanh":
            self.base_activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {base_activation}")
        
        # make grid points for splines
        grid = torch.linspace(
            grid_range[0], grid_range[1], 
            grid_size + 1, dtype=torch.float32
        )
        self.register_buffer("grid", grid)
        
        # spline weights
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # base weights for backup
        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """init all the weights"""
        with torch.no_grad():
            # random init for spline weights
            self.spline_weight.data.uniform_(-1/self.in_features, 1/self.in_features)
            
            # random init for base weights  
            self.base_weight.data.uniform_(-1/self.in_features, 1/self.in_features)
            
            # init spline scalers
            if self.enable_standalone_scale_spline:
                self.spline_scaler.data.fill_(self.scale_spline)
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        compute spline basis functions
        
        x: input (batch_size, in_features)
        returns: basis functions (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        batch_size = x.size(0)
        
        # simplified spline basis - using rbf-like functions
        grid = self.grid  # (grid_size + 1,)
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size + 1)
        
        # gaussian-like basis functions
        sigma = (grid[0, 0, 1] - grid[0, 0, 0]) / 2.0
        bases = torch.exp(-((x - grid) ** 2) / (2 * sigma ** 2))
        
        # pad to match expected size
        if bases.size(-1) < self.grid_size + self.spline_order:
            padding = self.grid_size + self.spline_order - bases.size(-1)
            bases = torch.cat([bases, torch.zeros(*bases.shape[:-1], padding, device=bases.device)], dim=-1)
        elif bases.size(-1) > self.grid_size + self.spline_order:
            bases = bases[..., :self.grid_size + self.spline_order]
        
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        convert curve samples to spline coefficients
        
        x: grid points
        y: function values  
        returns: spline coefficients
        """
        # simplified init - just reshape the noise
        return y.permute(2, 1, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through kan layer
        
        x: input (batch_size, in_features)
        returns: output (batch_size, out_features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        batch_size = x.size(0)
        
        # clamp input to grid range
        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        
        # compute base activation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # compute spline basis
        spline_basis = self.b_splines(x_clamped)  # (batch_size, in_features, grid_size + spline_order)
        
        # compute spline output
        # spline_basis: (batch_size, in_features, grid_size + spline_order)
        # spline_weight: (out_features, in_features, grid_size + spline_order)
        spline_output = torch.einsum(
            'big,oig->bo', spline_basis, self.spline_weight
        )
        
        if self.enable_standalone_scale_spline:
            spline_output = spline_output * self.spline_scaler.sum(dim=1, keepdim=True).T
        
        # mix base and spline outputs
        output = self.scale_base * base_output + self.scale_spline * spline_output
        
        return output
    
    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        """
        compute regularization loss for the layer
        
        regularize_activation: weight for activation reg
        regularize_entropy: weight for entropy reg
        returns: total reg loss
        """
        reg_loss = 0.0
        
        # l1 reg on spline weights
        if regularize_activation > 0:
            reg_loss += regularize_activation * torch.mean(torch.abs(self.spline_weight))
        
        # entropy reg (encourage sparsity)
        if regularize_entropy > 0:
            p = torch.softmax(torch.abs(self.spline_weight), dim=-1)
            entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
            reg_loss += regularize_entropy * torch.mean(entropy)
        
        return reg_loss 