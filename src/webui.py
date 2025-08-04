#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Predenergy Web Interface

This provides a web-based interface for interacting with Predenergy models.
"""

import os
import sys
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Predenergy.Predenergy import Predenergy


class PredenergyWebUI:
    def __init__(self):
        self.model = None
        self.model_path = None
    
    def load_model(self, model_path):
        """Load a trained Predenergy model"""
        try:
            self.model = Predenergy()
            self.model.load_model(model_path)
            self.model_path = model_path
            return f"Model loaded successfully from {model_path}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def predict(self, data_text, horizon=24):
        """Make predictions using the loaded model"""
        if self.model is None:
            return "Please load a model first", None
        
        try:
            # Parse input data
            data_lines = data_text.strip().split('\n')
            data_values = []
            for line in data_lines:
                if line.strip():
                    try:
                        value = float(line.strip())
                        data_values.append(value)
                    except ValueError:
                        continue
            
            if len(data_values) < self.model.seq_len:
                return f"Need at least {self.model.seq_len} data points", None
            
            # Create DataFrame
            data = pd.DataFrame(data_values, columns=['value'])
            
            # Make prediction
            predictions = self.model.predict(data, horizon=horizon)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot input data
            ax.plot(range(len(data_values)), data_values, 
                   label='Input Data', linewidth=2, color='blue')
            
            # Plot predictions
            pred_range = range(len(data_values), len(data_values) + len(predictions))
            ax.plot(pred_range, predictions.flatten(), 
                   label='Predictions', linewidth=2, color='red', linestyle='--')
            
            ax.set_title('Predenergy Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format predictions as text
            pred_text = "Predictions:\n" + "\n".join([f"{i+1}: {val:.4f}" 
                                                      for i, val in enumerate(predictions.flatten())])
            
            return pred_text, fig
            
        except Exception as e:
            return f"Error making prediction: {str(e)}", None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "No model loaded"
        
        try:
            info = self.model.get_model_info()
            info_text = "Model Information:\n"
            for key, value in info.items():
                info_text += f"{key}: {value}\n"
            return info_text
        except Exception as e:
            return f"Error getting model info: {str(e)}"


def create_ui():
    """Create the Gradio interface"""
    ui = PredenergyWebUI()
    
    with gr.Blocks(title="Predenergy Web Interface") as demo:
        gr.Markdown("# Predenergy Time Series Forecasting")
        gr.Markdown("Load a trained Predenergy model and make predictions on your time series data.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Model Loading")
                model_path_input = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to your trained model (.pth file)"
                )
                load_button = gr.Button("Load Model")
                load_output = gr.Textbox(label="Load Status")
                
                gr.Markdown("## Model Information")
                info_button = gr.Button("Get Model Info")
                info_output = gr.Textbox(label="Model Info")
            
            with gr.Column():
                gr.Markdown("## Prediction")
                data_input = gr.Textbox(
                    label="Input Data",
                    placeholder="Enter your time series data (one value per line)",
                    lines=10
                )
                horizon_input = gr.Slider(
                    minimum=1, maximum=100, value=24, step=1,
                    label="Prediction Horizon"
                )
                predict_button = gr.Button("Make Prediction")
                predict_output = gr.Textbox(label="Predictions")
                plot_output = gr.Plot(label="Prediction Plot")
        
        # Event handlers
        load_button.click(
            fn=ui.load_model,
            inputs=[model_path_input],
            outputs=[load_output]
        )
        
        info_button.click(
            fn=ui.get_model_info,
            inputs=[],
            outputs=[info_output]
        )
        
        predict_button.click(
            fn=ui.predict,
            inputs=[data_input, horizon_input],
            outputs=[predict_output, plot_output]
        )
    
    return demo


def main():
    """Main function to launch the web interface"""
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
