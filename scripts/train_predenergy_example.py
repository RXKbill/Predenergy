#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Training Example

This script demonstrates how to use the new configuration system for training Predenergy models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Predenergy import Predenergy
from models.Predenergy.utils.config_loader import load_predenergy_config


def create_sample_data(n_samples=1000, n_features=1):
    """Create sample time series data for demonstration"""
    np.random.seed(42)
    
    # 创建时间序列数据
    time_steps = np.arange(n_samples)
    trend = 0.1 * time_steps
    seasonality = 10 * np.sin(2 * np.pi * time_steps / 24)  # 24小时周期
    noise = np.random.normal(0, 1, n_samples)
    
    # 组合信号
    values = trend + seasonality + noise
    
    # 创建DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'value': values
    })
    
    return data


def main():
    """Main training example"""
    parser = argparse.ArgumentParser(description='Predenergy Training Example')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--use_combined_model', action='store_true',
                       help='Use combined Predenergy+MoTSE model')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (if not provided, will create sample data)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='predenergy_example',
                       help='Model name')
    
    args = parser.parse_args()
    
    print("=== Predenergy Training Example ===")
    
    # 1. 加载配置
    print("\n1. Loading configuration...")
    try:
        config = load_predenergy_config(
            config_path=args.config,
            use_combined_model=args.use_combined_model
        )
        print("Configuration loaded successfully")
        print(f"Model type: {'Combined' if args.use_combined_model else 'Standard'}")
        print(f"Sequence length: {config.seq_len}")
        print(f"Horizon: {config.horizon}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # 2. 准备数据
    print("\n2. Preparing data...")
    if args.data_path and os.path.exists(args.data_path):
        data = pd.read_csv(args.data_path)
        print(f"Loaded data from {args.data_path}")
    else:
        print("Creating sample data...")
        data = create_sample_data(n_samples=1000)
        print("Sample data created")
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    
    # 3. 初始化模型
    print("\n3. Initializing model...")
    try:
        model = Predenergy(
            use_combined_model=args.use_combined_model,
            **config.__dict__
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # 4. 设置数据加载器
    print("\n4. Setting up data loader...")
    try:
        model.setup_data_loader(
            data=data,
            batch_size=getattr(config, 'batch_size', 32),
            features=getattr(config, 'features', 'S'),
            target=getattr(config, 'target', 'value'),
            normalize=getattr(config, 'normalize', 2),
            train_ratio=getattr(config, 'train_ratio', 0.7),
            val_ratio=getattr(config, 'val_ratio', 0.2),
            shuffle=True
        )
        print("Data loader set up successfully")
    except Exception as e:
        print(f"Error setting up data loader: {e}")
        return
    
    # 5. 训练模型
    print("\n5. Training model...")
    try:
        model.fit(data, train_ratio_in_tv=getattr(config, 'train_ratio', 0.7))
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # 6. 保存模型
    print("\n6. Saving model...")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model_name}.pth")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # 7. 模型信息
    print("\n7. Model information:")
    print(model.get_model_summary())
    
    # 8. 训练历史
    history = model.get_training_history()
    if 'train_loss' in history and history['train_loss']:
        print(f"\nFinal training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    print("\n=== Training completed successfully! ===")


if __name__ == "__main__":
    main() 