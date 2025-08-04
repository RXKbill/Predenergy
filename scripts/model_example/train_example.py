#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Training Example

This script demonstrates how to train a Predenergy model using the PredenergyRunner.
"""

import os
from models.Predenergy.runner import PredenergyRunner
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig


def main():
    # Configuration
    model_config = PredenergyUnifiedConfig(
        input_size=1,
        hidden_size=512,
        seq_len=96,
        pred_len=24,
        label_len=48,
        c_out=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=5,
        distil=True,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu',
        output_attention=False,
    )
    
    # Save config
    os.makedirs('models/Predenergy_config', exist_ok=True)
    model_config.save_pretrained('models/Predenergy_config')
    
    # Training configuration
    train_config = {
        'model_path': 'models/Predenergy_config',
        'data_path': 'data/ETTh1.csv',  # Replace with your data path
        'batch_size': 32,
        'num_train_epochs': 10,
        'learning_rate': 1e-4,
        'precision': 'bf16',
        'seq_len': 96,
        'pred_len': 24,
        'features': 'S',
        'target': 'OT',
        'timeenc': 0,
        'freq': 'h',
        'normalize': 2,
    }
    
    # Initialize runner
    runner = PredenergyRunner(
        model_path='models/Predenergy_config',
        output_path='logs/Predenergy',
        seed=42
    )
    
    # Train model
    print("Starting Predenergy training...")
    model = runner.train_model(from_scratch=True, **train_config)
    print("Training completed!")


if __name__ == "__main__":
    main() 