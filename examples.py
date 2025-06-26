import torch
import numpy as np
import matplotlib.pyplot as plt
from kan_network import KAN
from kan_trainer import KANTrainer


def example_1d_function():
    """basic 1d function approximation example"""
    print("KAN 1D Function Approximation Example")
    print("-" * 40)
    
    # define target function
    def target_func(x):
        return torch.sin(5 * x) + 0.1 * torch.cos(20 * x)
    
    # Create KAN model
    model = KAN(layers_hidden=[1, 10, 1], grid_size=5)
    trainer = KANTrainer(model, optimizer_name="Adam", lr=0.01)
    
    # Generate training data
    x_train, y_train = trainer.create_dataset(
        func=target_func, n_samples=1000, input_dim=1, 
        noise_level=0.02, x_range=(-1, 1)
    )
    
    # Train model
    print("Training...")
    trainer.train(x_train, y_train, epochs=100, batch_size=64, verbose=False)
    
    # test and visualize
    x_test = torch.linspace(-1, 1, 200).unsqueeze(1)
    y_true = target_func(x_test)
    y_pred = trainer.predict(x_test)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_test.numpy(), y_true.numpy(), 'b-', label='True', linewidth=2)
    plt.plot(x_test.numpy(), y_pred.numpy(), 'r--', label='KAN', linewidth=2)
    plt.legend()
    plt.title('Function Approximation')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.training_history["train_loss"])
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    mse = torch.mean((y_true - y_pred) ** 2).item()
    print(f"Final MSE: {mse:.6f}")


def example_2d_function():
    """basic 2d function approximation example"""
    print("\nKAN 2D Function Approximation Example")
    print("-" * 40)
    
    # define 2d target function
    def target_func(x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.sin(x1) * torch.cos(x2)
    
    # Create KAN model
    model = KAN(layers_hidden=[2, 15, 1], grid_size=5)
    trainer = KANTrainer(model, optimizer_name="Adam", lr=0.01)
    
    # Generate training data
    x_train, y_train = trainer.create_dataset(
        func=target_func, n_samples=1500, input_dim=2,
        noise_level=0.01, x_range=(-2, 2)
    )
    
    # Train model
    print("Training...")
    trainer.train(x_train, y_train, epochs=150, batch_size=128, verbose=False)
    
    print(f"Final training loss: {trainer.training_history['train_loss'][-1]:.6f}")


if __name__ == "__main__":
    example_1d_function()
    example_2d_function()
    print("\nExamples completed!") 