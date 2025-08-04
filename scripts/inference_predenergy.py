#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Inference Script

This script provides a comprehensive inference interface for the Predenergy model.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Predenergy import Predenergy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predenergy Inference')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, 
                       help='Path to input data for prediction')
    parser.add_argument('--horizon', type=int, default=24, 
                       help='Prediction horizon')
    parser.add_argument('--output_path', type=str, default='predictions.csv', 
                       help='Output path for predictions')
    
    return parser.parse_args()


def load_data(data_path):
    """Load input data"""
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    return data


def main():
    """Main inference function"""
    args = parse_args()
    
    # Load model
    print("Loading model...")
    model = Predenergy()
    model.load_model(args.model_path)
    
    # Load data if provided
    if args.data_path:
        print("Loading input data...")
        data = load_data(args.data_path)
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(data, horizon=args.horizon)
        
        # Save predictions
        pred_df = pd.DataFrame(predictions, columns=['predicted_value'])
        pred_df.to_csv(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()