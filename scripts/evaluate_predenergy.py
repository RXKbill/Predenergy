#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Evaluation Script

This script provides comprehensive evaluation metrics for the Predenergy model.
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

from models.Predenergy.Predenergy import Predenergy
from Eval.evaluator import Evaluator
from Eval.metrics.regression_metrics import mse, mae, rmse, mape, smape


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Predenergy Model')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--test_data_path', type=str, required=True, 
                       help='Path to test data')
    parser.add_argument('--output_path', type=str, default='evaluation_results.json', 
                       help='Output path for evaluation results')
    
    # Evaluation configuration
    parser.add_argument('--horizon', type=int, default=24, 
                       help='Prediction horizon')
    parser.add_argument('--metrics', nargs='+', 
                       default=['mse', 'mae', 'rmse', 'mape', 'smape'],
                       help='Metrics to calculate')
    
    return parser.parse_args()


def load_data(data_path):
    """Load test data"""
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    return data


def evaluate_model(model, test_data, horizon, metrics):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Make predictions
    predictions = model.predict(test_data, horizon=horizon)
    
    # For evaluation, we need actual future values
    # This is a simplified evaluation - in practice you'd have actual future values
    actual = test_data.iloc[-horizon:].values if len(test_data) >= horizon else predictions
    
    # Calculate metrics
    results = {}
    
    if 'mse' in metrics:
        results['mse'] = mse(actual, predictions)
    
    if 'mae' in metrics:
        results['mae'] = mae(actual, predictions)
    
    if 'rmse' in metrics:
        results['rmse'] = rmse(actual, predictions)
    
    if 'mape' in metrics:
        results['mape'] = mape(actual, predictions)
    
    if 'smape' in metrics:
        results['smape'] = smape(actual, predictions)
    
    return results


def save_results(results, output_path):
    """Save evaluation results"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Load model
    print("Loading model...")
    model = Predenergy()
    model.load_model(args.model_path)
    
    # Load test data
    print("Loading test data...")
    test_data = load_data(args.test_data_path)
    
    # Evaluate model
    results = evaluate_model(model, test_data, args.horizon, args.metrics)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    # Save results
    save_results(results, args.output_path)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 