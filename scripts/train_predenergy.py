#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
Predenergy Training Script

This script provides a corrected training interface for the Predenergy model,
with proper configuration management and framework compatibility.
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import paddle
from paddle.io import DataLoader
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig, load_config_from_file
from models.Predenergy.models.modeling_Predenergy import PredenergyForPrediction
from models.Predenergy.utils.config_loader import ConfigLoader, create_config_from_args
from models.Predenergy.datasets.Predenergy_data_loader import create_Predenergy_data_loader
from models.utils import EarlyStopping
from utils.data_processing import split_before


def parse_args():
    """Parse command line arguments with corrected parameter handling"""
    parser = argparse.ArgumentParser(description='Train Predenergy Model')
    
    # Configuration file parameters
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (YAML/JSON)')
    
    # Model selection
    parser.add_argument('--use_combined_model', action='store_true', 
                       help='Use combined Predenergy+MoTSE model')
    
    # Core prediction parameters
    parser.add_argument('--seq_len', type=int, default=96, 
                       help='Input sequence length')
    parser.add_argument('--horizon', type=int, default=24, 
                       help='Prediction horizon')
    parser.add_argument('--label_len', type=int, default=48, 
                       help='Label length for decoder')
    parser.add_argument('--input_size', type=int, default=1,
                       help='Input feature dimension')
    parser.add_argument('--c_out', type=int, default=1,
                       help='Output dimension')
    
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=512, 
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, 
                       help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, 
                       help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, 
                       help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, 
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', 
                       help='Activation function')
    
    # Training parameters
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience')
    parser.add_argument('--loss_function', type=str, default='huber', 
                       choices=['huber', 'mse', 'mae'], 
                       help='Loss function')
    
    # Data processing parameters
    parser.add_argument('--features', type=str, default='S', 
                       choices=['S', 'M'], 
                       help='Feature type: S for univariate, M for multivariate')
    parser.add_argument('--target', type=str, default='OT', 
                       help='Target column name')
    parser.add_argument('--normalize', type=int, default=2, 
                       help='Normalization method')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                       help='Validation data ratio')
    
    # MoTSE specific parameters (for combined model)
    parser.add_argument('--num_experts', type=int, default=8,
                       help='Number of experts for MoTSE')
    parser.add_argument('--num_experts_per_tok', type=int, default=2,
                       help='Number of experts per token')
    parser.add_argument('--connection_type', type=str, default='adaptive',
                       choices=['linear', 'attention', 'concat', 'adaptive'],
                       help='Connection type between STDM and MoTSE')
    parser.add_argument('--motse_hidden_size', type=int, default=1024,
                       help='MoTSE hidden size')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='predenergy_model', 
                       help='Model name for saving')
    parser.add_argument('--save_config', action='store_true', 
                       help='Save model configuration')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    return parser.parse_args()


def setup_device_and_seed(args: argparse.Namespace) -> str:
    """Setup device and random seed"""
    # Set random seed
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    return device


def load_and_validate_config(args: argparse.Namespace) -> PredenergyUnifiedConfig:
    """Load and validate configuration"""
    loader = ConfigLoader()
    
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_file(args.config)
        
        # Override with command line arguments if provided
        args_dict = vars(args)
        parser = argparse.ArgumentParser()
        for action in parser._actions:
            if action.dest != 'help':
                parser.set_defaults(**{action.dest: action.default})
        for key, value in args_dict.items():
            if hasattr(config, key):
                default_value = parser.get_default(key)
                if value != default_value:
                    setattr(config, key, value)
                    print(f"Overriding config.{key} = {value}")
    else:
        print("Creating configuration from command line arguments")
        config = create_config_from_args(vars(args))
    
    # Validate configuration
    if not loader.validate_config(config):
        raise ValueError("Invalid configuration")
    
    print("\n" + loader.get_config_summary(config))
    return config


def load_data(data_path: str, config: PredenergyUnifiedConfig) -> tuple:
    """Load and split data"""
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported data format. Use CSV or Parquet files.")
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    
    # Validate target column
    if config.target not in data.columns:
        raise ValueError(f"Target column '{config.target}' not found in data")
    
    # Create data loaders
    data_loader = create_Predenergy_data_loader(
        data=data,
        seq_len=config.seq_len,
        pred_len=config.horizon,
        batch_size=config.batch_size,
        features=config.features,
        target=config.target,
        normalize=config.normalize,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        loader_type='standard'
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, data_loader


def create_model(config: PredenergyUnifiedConfig, device: str) -> PredenergyForPrediction:
    """Create and initialize model"""
    print("Creating model...")
    
    try:
        model = PredenergyForPrediction(config)
        if device == 'gpu':
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        raise


def setup_training_components(model: PredenergyForPrediction, config: PredenergyUnifiedConfig):
    """Setup optimizer, scheduler, and early stopping"""
    print("Setting up training components...")
    
    # Optimizer
    optimizer = paddle.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = paddle.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.patience // 2,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        delta=1e-6
    )
    
    return optimizer, scheduler, early_stopping


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Move batch to device
        if isinstance(batch, dict):
            input_data = batch['input_ids'].to(device)
            labels = batch.get('labels', batch.get('target')).to(device)
            loss_masks = batch.get('loss_mask', None)
            if loss_masks is not None:
                loss_masks = loss_masks.to(device)
        else:
            # Handle tuple format
            input_data, labels = batch[0].to(device), batch[1].to(device)
            loss_masks = None
        
        # Forward pass
        outputs = model(
            input_data=input_data,
            labels=labels,
            loss_masks=loss_masks,
            return_dict=True
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with paddle.no_grad():
        for batch in val_loader:
            # Move batch to device
            if isinstance(batch, dict):
                input_data = batch['input_ids'].to(device)
                labels = batch.get('labels', batch.get('target')).to(device)
                loss_masks = batch.get('loss_mask', None)
                if loss_masks is not None:
                    loss_masks = loss_masks.to(device)
            else:
                # Handle tuple format
                input_data, labels = batch[0].to(device), batch[1].to(device)
                loss_masks = None
            
            # Forward pass
            outputs = model(
                input_data=input_data,
                labels=labels,
                loss_masks=loss_masks,
                return_dict=True
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_model_and_config(model, config, args, epoch, val_loss):
    """Save model and configuration"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{args.model_name}_epoch_{epoch}.pth"
    paddle.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict()
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Save configuration
    if args.save_config:
        config_path = output_dir / f"{args.model_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {config_path}")


def main():
    """Main training function"""
    print("=== Predenergy Training Script ===")
    
    # Parse arguments
    args = parse_args()
    
    # Setup device and seed
    device = setup_device_and_seed(args)
    
    # Load and validate configuration
    config = load_and_validate_config(args)
    config.device = device
    
    # Load data
    train_loader, val_loader, test_loader, data_loader = load_data(args.data_path, config)
    
    # Create model
    model = create_model(config, device)
    
    # Setup training components
    optimizer, scheduler, early_stopping = setup_training_components(model, config)
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_and_config(model, config, args, epoch + 1, val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    
    # Final test
    if test_loader:
        print("Running final test evaluation...")
        test_loss = validate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise