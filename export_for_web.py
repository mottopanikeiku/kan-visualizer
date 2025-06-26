import torch
import json
import numpy as np
from kan_network import KAN
from kan_trainer import KANTrainer


def export_kan_for_visualization(model, trainer=None, save_path="web/data/model.json"):
    """
    export trained kan model to json format for web visualization
    
    model: trained kan model
    trainer: kan trainer (optional, for training history)
    save_path: where to save the json file
    """
    
    export_data = {
        'metadata': {
            'architecture': model.layers_hidden,
            'grid_size': model.grid_size,
            'spline_order': model.spline_order,
            'num_layers': len(model.layers),
            'total_parameters': sum(p.numel() for p in model.parameters())
        },
        'layers': [],
        'training_history': trainer.training_history if trainer else None
    }
    
    # export each layer
    for i, layer in enumerate(model.layers):
        # evaluate spline functions on fine grid for visualization
        x_fine = torch.linspace(-2, 2, 200)
        spline_evaluations = []
        
        with torch.no_grad():
            layer.eval()
            for out_idx in range(layer.out_features):
                for in_idx in range(layer.in_features):
                                         # evaluate this specific spline function - skip for now
                     pass
                    
                                         # get individual spline evaluations
                     individual_evaluations = []
                     for x_val in x_fine:
                         x_input = torch.zeros(1, layer.in_features)
                         x_input[0, in_idx] = x_val
                         
                         # simplified evaluation - just use spline coefficients directly
                         value = 0.0
                         if len(layer.spline_weight[out_idx, in_idx]) > 0:
                             # use middle coefficient as approximation
                             mid_idx = len(layer.spline_weight[out_idx, in_idx]) // 2
                             value = float(layer.spline_weight[out_idx, in_idx, mid_idx]) * float(x_val)
                         
                         individual_evaluations.append(value)
                     
                     spline_evaluations.append({
                         'input_idx': in_idx,
                         'output_idx': out_idx,
                         'x_values': x_fine.tolist(),
                         'y_values': individual_evaluations
                     })
        
        layer_data = {
            'layer_index': i,
            'input_features': layer.in_features,
            'output_features': layer.out_features,
            'grid_points': layer.grid.tolist(),
            'grid_range': [float(layer.grid_range[0]), float(layer.grid_range[1])],
            'spline_coefficients': layer.spline_weight.detach().tolist(),
            'base_weights': layer.base_weight.detach().tolist(),
            'scale_base': float(layer.scale_base),
            'scale_spline': float(layer.scale_spline),
            'spline_evaluations': spline_evaluations
        }
        
        export_data['layers'].append(layer_data)
    
    # save to json file
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"kan model exported to {save_path}")
    return export_data


def create_sample_datasets_for_web():
    """create sample datasets that will be used in web visualization"""
    
    # simple 1d function
    def func_1d(x):
        return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)
    
    # 2d function
    def func_2d(x):
        return torch.sin(x[:, 0]) * torch.exp(-x[:, 1]**2)
    
    datasets = {
        '1d_sine_wave': {
            'name': '1D Sine Wave',
            'description': 'sin(3x) + 0.3*cos(10x)',
            'input_dim': 1,
            'function': 'Math.sin(3*x) + 0.3*Math.cos(10*x)',
            'samples': []
        },
        '2d_gaussian': {
            'name': '2D Gaussian Wave',
            'description': 'sin(x) * exp(-y²)',
            'input_dim': 2,
            'function': 'Math.sin(x) * Math.exp(-y*y)',
            'samples': []
        }
    }
    
    # generate sample points for 1d function
    x_1d = torch.linspace(-2, 2, 100).unsqueeze(1)
    y_1d = func_1d(x_1d)
    
    for i in range(len(x_1d)):
        datasets['1d_sine_wave']['samples'].append({
            'input': [float(x_1d[i, 0])],
            'output': float(y_1d[i, 0])
        })
    
    # generate sample points for 2d function
    x1 = torch.linspace(-2, 2, 30)
    x2 = torch.linspace(-2, 2, 30)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_2d = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    y_2d = func_2d(x_2d)
    
    for i in range(len(x_2d)):
        datasets['2d_gaussian']['samples'].append({
            'input': [float(x_2d[i, 0]), float(x_2d[i, 1])],
            'output': float(y_2d[i])
        })
    
    # save datasets
    with open('web/data/datasets.json', 'w') as f:
        json.dump(datasets, f, indent=2)
    
    print("sample datasets exported to web/data/datasets.json")
    return datasets


def train_and_export_sample_models():
    """train sample models and export them for web visualization"""
    
    print("training sample kan models for web visualization...")
    
    # 1d function approximation
    print("\n1. training 1d function approximation...")
    model_1d = KAN(layers_hidden=[1, 10, 1], grid_size=5)
    trainer_1d = KANTrainer(model_1d, optimizer_name="Adam", lr=0.01)
    
    def target_1d(x):
        return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)
    
    x_train_1d, y_train_1d = trainer_1d.create_dataset(
        func=target_1d, n_samples=1000, input_dim=1, x_range=(-2, 2)
    )
    
    trainer_1d.train(x_train_1d, y_train_1d, epochs=100, batch_size=64, verbose=False)
    export_kan_for_visualization(model_1d, trainer_1d, "web/data/model_1d.json")
    
    # 2d function approximation
    print("2. training 2d function approximation...")
    model_2d = KAN(layers_hidden=[2, 15, 1], grid_size=5)
    trainer_2d = KANTrainer(model_2d, optimizer_name="Adam", lr=0.01)
    
    def target_2d(x):
        return torch.sin(x[:, 0]) * torch.exp(-x[:, 1]**2)
    
    x_train_2d, y_train_2d = trainer_2d.create_dataset(
        func=target_2d, n_samples=1500, input_dim=2, x_range=(-2, 2)
    )
    
    trainer_2d.train(x_train_2d, y_train_2d, epochs=150, batch_size=128, verbose=False)
    export_kan_for_visualization(model_2d, trainer_2d, "web/data/model_2d.json")
    
    # complex network
    print("3. training complex network...")
    model_complex = KAN(layers_hidden=[2, 20, 15, 10, 1], grid_size=7)
    trainer_complex = KANTrainer(model_complex, optimizer_name="Adam", lr=0.005)
    
    def target_complex(x):
        return torch.sin(x[:, 0] * x[:, 1]) + 0.5 * torch.tanh(x[:, 0] - x[:, 1])
    
    x_train_complex, y_train_complex = trainer_complex.create_dataset(
        func=target_complex, n_samples=2000, input_dim=2, x_range=(-2, 2)
    )
    
    trainer_complex.train(x_train_complex, y_train_complex, epochs=200, batch_size=128, verbose=False)
    export_kan_for_visualization(model_complex, trainer_complex, "web/data/model_complex.json")
    
    # create datasets
    create_sample_datasets_for_web()
    
    print("\nall sample models exported successfully!")
    print("files created:")
    print("- web/data/model_1d.json")
    print("- web/data/model_2d.json") 
    print("- web/data/model_complex.json")
    print("- web/data/datasets.json")


if __name__ == "__main__":
    train_and_export_sample_models() 