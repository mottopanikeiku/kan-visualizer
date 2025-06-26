import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from kan_layer import KANLayer


class KAN(nn.Module):
    """
    kolmogorov-arnold network
    
    neural net that uses learnable functions instead of boring linear layers.
    way cooler than regular mlps.
    """
    
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: str = "silu",
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        """
        init kan network
        
        layers_hidden: list of layer sizes [input, hidden1, hidden2, ..., output]
        grid_size: number of grid points for splines
        spline_order: spline order (3 = cubic)
        scale_noise: noise for init
        scale_base: base activation weight
        scale_spline: spline activation weight  
        base_activation: base function ("silu", "relu", "gelu", "tanh")
        grid_eps: tiny epsilon for stability
        grid_range: input normalization range
        """
        super(KAN, self).__init__()
        
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KANLayer(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        """
        forward pass through kan network
        
        x: input tensor (batch_size, input_dim)
        update_grid: whether to update grid points
        returns: output tensor (batch_size, output_dim)
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(
        self, 
        regularize_activation: float = 1.0, 
        regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """
        compute total reg loss across all layers
        
        regularize_activation: weight for activation reg
        regularize_entropy: weight for entropy reg  
        returns: total reg loss
        """
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return reg_loss
    
    def get_subset(self, in_id: List[int], out_id: List[int]) -> 'KAN':
        """
        extract subset of network
        
        in_id: input indices to keep
        out_id: output indices to keep
        returns: new kan with subset of connections
        """
        # would need to create new kan with pruned connections
        # depends on specific pruning strategy
        raise NotImplementedError("Subset extraction not implemented")
    
    def prune(self, threshold: float = 0.01) -> 'KAN':
        """
        prune network by removing weak connections
        
        threshold: cutoff for removing connections
        returns: pruned network
        """
        # would analyze spline weights and remove weak connections
        raise NotImplementedError("Pruning not implemented")
    
    def symbolic_formula(self, var_name: str = "x") -> str:
        """
        extract symbolic formula (when network is simple enough)
        
        var_name: variable name for formula
        returns: symbolic representation of learned function
        """
        # would analyze learned splines and convert to symbolic expressions
        raise NotImplementedError("Symbolic formula extraction not implemented")


class MultilayerKAN(nn.Module):
    """
    extended kan with different configs per layer
    """
    
    def __init__(self, layer_configs: List[dict]):
        """
        init multilayer kan with per-layer configs
        
        layer_configs: list of dicts with kan layer params
        """
        super(MultilayerKAN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for config in layer_configs:
            self.layers.append(KANLayer(**config))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through all layers"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def regularization_loss(
        self, 
        regularize_activation: float = 1.0, 
        regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """compute total reg loss"""
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return reg_loss 