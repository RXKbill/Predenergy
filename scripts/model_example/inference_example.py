#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Inference Example

This script demonstrates how to use a trained Predenergy model for inference.
"""

import torch
import numpy as np
from models.Predenergy.runner import PredenergyRunner
from models.Predenergy.models.modeling_Predenergy import PredenergyForPrediction


def main():
    # Load trained model
    model_path = 'logs/Predenergy'  # Path to your trained model
    runner = PredenergyRunner(model_path=model_path)
    
    # Load model
    model = runner.load_model(model_path)
    model.eval()
    
    # Example input data (replace with your actual data)
    batch_size = 1
    seq_len = 96
    pred_len = 24
    
    # Create dummy input data
    x_enc = torch.randn(batch_size, seq_len, 1)  # [batch_size, seq_len, input_size]
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # [batch_size, seq_len, time_features]
    x_dec = torch.randn(batch_size, pred_len, 1)  # [batch_size, pred_len, output_size]
    x_mark_dec = torch.randn(batch_size, pred_len, 4)  # [batch_size, pred_len, time_features]
    
    # Run inference
    with torch.no_grad():
        predictions = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions: {predictions}")


def predict_from_data(data_path: str, model_path: str):
    """Predict using actual data"""
    runner = PredenergyRunner(model_path=model_path)
    
    # Configuration for prediction
    pred_config = {
        'seq_len': 96,
        'pred_len': 24,
        'features': 'S',
        'target': 'OT',
        'timeenc': 0,
        'freq': 'h',
        'normalize': 2,
    }
    
    # Run prediction
    predictions = runner.predict(data_path, model_path, **pred_config)
    
    print(f"Predictions shape: {predictions.shape}")
    return predictions


if __name__ == "__main__":
    # Example with dummy data
    main()
    
    # Example with actual data (uncomment if you have data)
    # data_path = 'data/ETTh1.csv'
    # model_path = 'logs/Predenergy'
    # predictions = predict_from_data(data_path, model_path) 