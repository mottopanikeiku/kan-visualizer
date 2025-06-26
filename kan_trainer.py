import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Callable
from tqdm import tqdm
import os

from kan_network import KAN


class KANTrainer:
    """
    trainer for kan networks
    """
    
    def __init__(
        self,
        model: KAN,
        device: Optional[torch.device] = None,
        optimizer_name: str = "LBFGS",
        lr: float = 1.0,
        weight_decay: float = 0.0,
    ):
        """
        init kan trainer
        
        model: kan model to train
        device: where to train (cpu/gpu)
        optimizer_name: optimizer type ("LBFGS", "Adam", "AdamW")  
        lr: learning rate
        weight_decay: weight decay for reg
        """
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # setup optimizer
        if optimizer_name == "LBFGS":
            self.optimizer = optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=20,
                tolerance_grad=1e-8,
                tolerance_change=1e-12,
                history_size=100
            )
        elif optimizer_name == "Adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.optimizer_name = optimizer_name
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "reg_loss": []
        }
    
    def create_dataset(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        n_samples: int = 1000,
        input_dim: int = 1,
        noise_level: float = 0.0,
        x_range: Tuple[float, float] = (-1, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        create synthetic dataset from function
        
        func: function to approximate
        n_samples: number of samples
        input_dim: input dimensions
        noise_level: noise to add to outputs
        x_range: range for input sampling
        returns: (inputs, targets)
        """
        x = torch.rand(n_samples, input_dim) * (x_range[1] - x_range[0]) + x_range[0]
        y = func(x)
        
        if noise_level > 0:
            y += torch.randn_like(y) * noise_level
        
        return x, y
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        regularize_activation: float = 0.0,
        regularize_entropy: float = 0.0
    ) -> float:
        """
        single training step
        
        x_batch: input batch
        y_batch: target batch 
        regularize_activation: activation reg weight
        regularize_entropy: entropy reg weight
        returns: loss value
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        def closure():
            self.optimizer.zero_grad()
            
            # forward pass
            y_pred = self.model(x_batch)
            
            # compute losses
            mse_loss = nn.MSELoss()(y_pred, y_batch)
            reg_loss = self.model.regularization_loss(regularize_activation, regularize_entropy)
            
            total_loss = mse_loss + reg_loss
            total_loss.backward()
            
            return total_loss
        
        if self.optimizer_name == "LBFGS":
            loss = self.optimizer.step(closure)
            return loss.item()
        else:
            loss = closure()
            self.optimizer.step()
            return loss.item()
    
    def evaluate(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> float:
        """
        evaluate model on validation data
        
        x_val: validation inputs
        y_val: validation targets
        returns: validation loss
        """
        self.model.eval()
        with torch.no_grad():
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            
            y_pred = self.model(x_val)
            val_loss = nn.MSELoss()(y_pred, y_val)
            
        self.model.train()
        return val_loss.item()
    
    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        regularize_activation: float = 0.0,
        regularize_entropy: float = 0.0,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        train the kan model
        
        x_train: training inputs
        y_train: training targets
        x_val: validation inputs
        y_val: validation targets
        epochs: number of training epochs
        batch_size: batch size (if none, use full batch)
        regularize_activation: activation reg weight
        regularize_entropy: entropy reg weight
        save_path: path to save best model
        verbose: whether to print progress
        returns: training history dict
        """
        # use full batch if batch_size not specified
        if batch_size is None:
            batch_size = len(x_train)
        
        # create data loader
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        
        self.model.train()
        
        for epoch in tqdm(range(epochs), disable=not verbose):
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            
            for x_batch, y_batch in dataloader:
                # training step
                loss = self.train_step(
                    x_batch, y_batch,
                    regularize_activation, regularize_entropy
                )
                epoch_loss += loss
                
                # compute reg loss for logging
                with torch.no_grad():
                    reg_loss = self.model.regularization_loss(
                        regularize_activation, regularize_entropy
                    )
                    if isinstance(reg_loss, torch.Tensor):
                        epoch_reg_loss += reg_loss.item()
                    else:
                        epoch_reg_loss += reg_loss
            
            # average losses over batches
            avg_train_loss = epoch_loss / len(dataloader)
            avg_reg_loss = epoch_reg_loss / len(dataloader)
            
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["reg_loss"].append(avg_reg_loss)
            
            # validation
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.training_history["val_loss"].append(val_loss)
                
                # save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"Reg Loss: {avg_reg_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Reg Loss: {avg_reg_loss:.6f}")
        
        return self.training_history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.training_history["train_loss"], label="Train Loss")
        if self.training_history["val_loss"]:
            axes[0].plot(self.training_history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training/Validation Loss")
        axes[0].legend()
        axes[0].set_yscale("log")
        
        # Regularization loss plot
        axes[1].plot(self.training_history["reg_loss"], label="Reg Loss", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Regularization Loss")
        axes[1].set_title("Regularization Loss")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model
        
        Args:
            x: Input tensor
        
        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            predictions = self.model(x)
        return predictions.cpu()
    
    def save_model(self, path: str):
        """Save model state dict"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state dict"""
        self.model.load_state_dict(torch.load(path, map_location=self.device)) 