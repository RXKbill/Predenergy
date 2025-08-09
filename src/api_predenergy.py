"""
Predenergy API Server
This module provides a RESTful API for the Predenergy time series forecasting model.
"""

import os
import sys
import asyncio
import json
import numpy as np
import pandas as pd
import paddle
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Predenergy.Predenergy import Predenergy
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    data: List[float] = Field(..., description="Input time series data")
    horizon: int = Field(24, description="Prediction horizon", ge=1, le=365)
    model_config: Optional[Dict[str, Any]] = Field(None, description="Optional model configuration overrides")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float] = Field(..., description="Forecast predictions")
    horizon: int = Field(..., description="Prediction horizon")
    input_length: int = Field(..., description="Length of input data")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_info: Dict[str, Any] = Field(..., description="Comprehensive model information")
    model_path: str = Field(..., description="Current model path")
    status: str = Field(..., description="Model status")


class TrainingRequest(BaseModel):
    """Request model for training"""
    config: Dict[str, Any] = Field(..., description="Training configuration")
    data_path: Optional[str] = Field(None, description="Path to training data")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    paddle_available: bool = Field(..., description="Whether PaddlePaddle is available")
    gpu_available: bool = Field(..., description="Whether GPU is available")


class PredenergyAPI:
    """
    Predenergy API server with proper error handling and configuration management.
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Predenergy API",
            description="RESTful API for Predenergy time series forecasting",
            version="2.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.model: Optional[Predenergy] = None
        self.model_path: Optional[str] = None
        self.config: Optional[PredenergyUnifiedConfig] = None
        
        self.setup_routes()
        
        print("Predenergy API initialized")
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information"""
            return {
                "message": "Predenergy API",
                "version": "2.1.0",
                "status": "running",
                "framework": "PaddlePaddle",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None and self.model.is_fitted,
                paddle_available=True,
                gpu_available=paddle.device.cuda.device_count() > 0
            )
        
        @self.app.post("/load_model")
        async def load_model(model_path: str, config_overrides: Optional[Dict[str, Any]] = None):
            """Load a trained model from file"""
            try:
                if not os.path.exists(model_path):
                    raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
                
                # Initialize model with default config
                self.model = Predenergy()
                
                # Apply config overrides if provided
                if config_overrides:
                    for key, value in config_overrides.items():
                        if hasattr(self.model.config, key):
                            setattr(self.model.config, key, value)
                
                # Load the model
                self.model.load_model(model_path)
                self.model_path = model_path
                self.config = self.model.config
                
                return {
                    "message": "Model loaded successfully",
                    "model_path": model_path,
                    "model_info": self.model.get_model_info()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        @self.app.post("/create_model")
        async def create_model(config: Dict[str, Any]):
            """Create a new model with the given configuration"""
            try:
                # Create unified configuration
                self.config = PredenergyUnifiedConfig.from_dict(config)
                
                # Create model
                self.model = Predenergy(**config)
                
                return {
                    "message": "Model created successfully",
                    "model_info": self.model.get_model_info()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")
        
        @self.app.get("/model_info", response_model=ModelInfoResponse)
        async def get_model_info():
            """Get information about the current model"""
            if self.model is None:
                raise HTTPException(status_code=404, detail="No model loaded")
            
            return ModelInfoResponse(
                model_info=self.model.get_model_info(),
                model_path=self.model_path or "Not saved",
                status="fitted" if self.model.is_fitted else "not_fitted"
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make predictions using the loaded model"""
            if self.model is None:
                raise HTTPException(status_code=404, detail="No model loaded")
            
            if not self.model.is_fitted:
                raise HTTPException(status_code=400, detail="Model is not fitted")
            
            try:
                # Validate input data
                if len(request.data) < self.model.config.seq_len:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Input data length ({len(request.data)}) must be at least {self.model.config.seq_len}"
                    )
                
                # Prepare input data
                input_data = np.array(request.data)
                
                # Take the last seq_len points if input is longer
                if len(input_data) > self.model.config.seq_len:
                    input_data = input_data[-self.model.config.seq_len:]
                
                # Convert to DataFrame for compatibility
                df = pd.DataFrame(input_data, columns=[self.model.config.target])
                
                # Make prediction
                predictions = self.model.forecast(
                    horizon=request.horizon,
                    series=df
                )
                
                # Convert predictions to list
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                
                predictions_list = predictions.tolist()
                
                return PredictionResponse(
                    predictions=predictions_list,
                    horizon=request.horizon,
                    input_length=len(request.data),
                    model_info={
                        "model_type": "Combined" if self.model.config.use_combined_model else "Standard",
                        "seq_len": self.model.config.seq_len,
                        "horizon": self.model.config.horizon
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.post("/predict_file")
        async def predict_from_file(
            file: UploadFile = File(...),
            horizon: int = 24,
            target_column: Optional[str] = None
        ):
            """Make predictions from uploaded CSV file"""
            if self.model is None:
                raise HTTPException(status_code=404, detail="No model loaded")
            
            if not self.model.is_fitted:
                raise HTTPException(status_code=400, detail="Model is not fitted")
            
            try:
                # Validate file type
                if not file.filename.endswith('.csv'):
                    raise HTTPException(status_code=400, detail="Only CSV files are supported")
                
                # Read file
                contents = await file.read()
                df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
                
                # Determine target column
                if target_column is None:
                    target_column = self.model.config.target
                
                if target_column not in df.columns:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Target column '{target_column}' not found in file"
                    )
                
                # Extract data
                data = df[target_column].values
                
                # Validate data length
                if len(data) < self.model.config.seq_len:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Data length ({len(data)}) must be at least {self.model.config.seq_len}"
                    )
                
                # Take the last seq_len points
                input_data = data[-self.model.config.seq_len:]
                
                # Create DataFrame
                input_df = pd.DataFrame(input_data, columns=[target_column])
                
                # Make prediction
                predictions = self.model.forecast(
                    horizon=horizon,
                    series=input_df
                )
                
                # Convert predictions to list
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                
                predictions_list = predictions.tolist()
                
                return {
                    "predictions": predictions_list,
                    "horizon": horizon,
                    "input_length": len(input_data),
                    "source_file": file.filename,
                    "target_column": target_column,
                    "model_info": {
                        "model_type": "Combined" if self.model.config.use_combined_model else "Standard",
                        "seq_len": self.model.config.seq_len
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")
        
        @self.app.post("/train")
        async def train_model(request: TrainingRequest):
            """Train the model (basic implementation)"""
            if self.model is None:
                raise HTTPException(status_code=404, detail="No model created. Use /create_model first")
            
            try:
                # Load training data
                if request.data_path and os.path.exists(request.data_path):
                    train_data = pd.read_csv(request.data_path)
                else:
                    raise HTTPException(status_code=400, detail="Valid data_path required for training")
                
                # Setup data loader
                self.model.setup_data_loader(
                    data=train_data,
                    **request.config.get('data_loader', {})
                )
                
                # Train the model
                self.model.forecast_fit(
                    train_valid_data=train_data,
                    **request.config.get('training', {})
                )
                
                return {
                    "message": "Model training completed",
                    "model_info": self.model.get_model_info()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
        
        @self.app.post("/save_model")
        async def save_model(save_path: str):
            """Save the current model"""
            if self.model is None:
                raise HTTPException(status_code=404, detail="No model to save")
            
            try:
                # Create directory if it doesn't exist
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save model
                self.model.save_model(save_path)
                self.model_path = save_path
                
                return {
                    "message": "Model saved successfully",
                    "save_path": save_path
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        print(f"Starting Predenergy API server on {host}:{port}")
        print(f"API documentation available at: http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs
        )


def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predenergy API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model_path", type=str, help="Path to pre-load a model")
    
    args = parser.parse_args()
    
    # Create API instance
    api = PredenergyAPI()
    
    # Pre-load model if specified
    if args.model_path:
        try:
            print(f"Pre-loading model from: {args.model_path}")
            # This would need to be done asynchronously in a real implementation
            # For now, we'll just note the path
            api.model_path = args.model_path
        except Exception as e:
            print(f"Warning: Failed to pre-load model: {e}")
    
    # Run server
    api.run(
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()