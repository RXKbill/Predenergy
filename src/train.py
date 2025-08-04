#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Training Entry Point

This is the main entry point for training Predenergy models.
It provides a simplified interface for training with configuration files.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Predenergy.Predenergy import Predenergy
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig


def load_config(config_path):
    """Load configuration from file"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def create_model_from_config(config):
    """Create model from configuration"""
    use_combined_model = config.get('use_combined_model', False)
    
    # Handle horizon/pred_len mapping
    if 'horizon' in config and 'pred_len' not in config:
        config['pred_len'] = config['horizon']
    
    # Set default values for missing parameters
    if not use_combined_model:
        # Standard Predenergy defaults
        config.setdefault('label_len', config.get('seq_len', 96) // 2)
        config.setdefault('enc_in', 1)
        config.setdefault('dec_in', 1)
        config.setdefault('c_out', 1)
        config.setdefault('input_size', 1)
        config.setdefault('hidden_size', 512)
        config.setdefault('factor', 5)
        config.setdefault('moving_avg', 25)
        config.setdefault('distil', True)
        config.setdefault('embed', 'timeF')
        config.setdefault('output_attention', False)
        config.setdefault('patch_len', 16)
        config.setdefault('stride', 8)
        config.setdefault('period_len', 4)
        config.setdefault('seg_len', 6)
        config.setdefault('win_size', 2)
        config.setdefault('fc_dropout', 0.2)
        config.setdefault('num_experts', 4)
        config.setdefault('noisy_gating', True)
        config.setdefault('k', 1)
        config.setdefault('CI', True)
    else:
        # Combined model defaults
        config.setdefault('label_len', config.get('seq_len', 96) // 2)
        config.setdefault('enc_in', 1)
        config.setdefault('dec_in', 1)
        config.setdefault('c_out', 1)
        config.setdefault('input_size', 1)
        config.setdefault('hidden_size', 512)
        config.setdefault('factor', 5)
        config.setdefault('moving_avg', 25)
        config.setdefault('distil', True)
        config.setdefault('embed', 'timeF')
        config.setdefault('output_attention', False)
        config.setdefault('patch_len', 16)
        config.setdefault('stride', 8)
        config.setdefault('period_len', 4)
        config.setdefault('seg_len', 6)
        config.setdefault('win_size', 2)
        config.setdefault('fc_dropout', 0.2)
        config.setdefault('num_experts', 4)
        config.setdefault('noisy_gating', True)
        config.setdefault('k', 1)
        config.setdefault('CI', True)
        config.setdefault('connection_type', 'adaptive')
        config.setdefault('connection_dropout', 0.1)
        config.setdefault('connection_hidden_size', 256)
        config.setdefault('use_cross_attention', True)
        config.setdefault('use_residual_connection', True)
        config.setdefault('use_layer_norm', True)
        config.setdefault('apply_aux_loss', True)
        config.setdefault('router_aux_loss_factor', 0.02)
        
        # MoTSE defaults
        config.setdefault('motse_hidden_size', 512)
        config.setdefault('motse_intermediate_size', 1024)
        config.setdefault('motse_num_hidden_layers', 4)
        config.setdefault('motse_num_attention_heads', 8)
        config.setdefault('motse_num_key_value_heads', 8)
        config.setdefault('motse_hidden_act', 'silu')
        config.setdefault('motse_num_experts_per_tok', 2)
        config.setdefault('motse_max_position_embeddings', 2048)
        config.setdefault('motse_initializer_range', 0.02)
        config.setdefault('motse_rms_norm_eps', 1e-6)
        config.setdefault('motse_use_cache', True)
        config.setdefault('motse_use_dense', False)
        config.setdefault('motse_rope_theta', 10000)
        config.setdefault('motse_attention_dropout', 0.0)
        config.setdefault('motse_apply_aux_loss', True)
        config.setdefault('motse_router_aux_loss_factor', 0.02)
        config.setdefault('motse_tie_word_embeddings', False)
        config.setdefault('motse_input_size', 1)
        config.setdefault('motse_horizon_lengths', [24])
        config.setdefault('motse_attn_implementation', 'eager')
    
    # Training defaults
    config.setdefault('batch_size', 32)
    config.setdefault('num_epochs', 100)
    config.setdefault('lr', 0.001)
    config.setdefault('patience', 10)
    config.setdefault('loss', 'huber')
    config.setdefault('lradj', 'type3')
    config.setdefault('num_workers', 0)
    
    # Data defaults
    config.setdefault('features', 'S')
    config.setdefault('target', 'OT')
    config.setdefault('freq', 'h')
    config.setdefault('normalize', 2)
    config.setdefault('train_ratio', 0.7)
    config.setdefault('val_ratio', 0.2)
    
    if use_combined_model:
        model_config = PredenergyUnifiedConfig(**config)
    else:
        model_config = PredenergyUnifiedConfig(**config)
    
    model = Predenergy(use_combined_model=use_combined_model, **config)
    return model


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Predenergy Model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    
    # Load data
    print("Loading data...")
    import pandas as pd
    data = pd.read_csv(args.data_path)
    
    # Setup data loader
    print("Setting up data loader...")
    model.setup_data_loader(
        data=data,
        batch_size=config.get('batch_size', 32),
        features=config.get('features', 'S'),
        target=config.get('target', 'OT'),
        freq=config.get('freq', 'h'),
        normalize=config.get('normalize', 2),
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.2),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    # Train model
    print("Starting training...")
    model.fit(data, train_ratio_in_tv=config.get('train_ratio', 0.7))
    
    # Save model
    model_path = os.path.join(args.output_dir, 'predenergy_model.pth')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()