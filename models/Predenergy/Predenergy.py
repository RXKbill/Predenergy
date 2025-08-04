"""
Predenergy Main Model Class
This module provides a corrected main Predenergy model class with proper framework compatibility.
"""

import os
import paddle
import paddle.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import warnings
from pathlib import Path

from models.unified_config import PredenergyUnifiedConfig
from models.modeling_Predenergy import PredenergyForPrediction
from datasets.Predenergy_data_loader import create_Predenergy_data_loader
from utils.config_loader import ConfigLoader, load_predenergy_config
from models.model_base import ModelBase


class Predenergy(ModelBase):
    """
    Predenergy model class with proper framework compatibility and unified configuration.
    This class serves as the main interface for the Predenergy model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Predenergy model.
        
        Args:
            **kwargs: Configuration parameters
        """
        super(Predenergy, self).__init__()
        
        # Load configuration
        config_path = kwargs.get('config_path', None)
        if config_path:
            self.config = load_predenergy_config(config_path)
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.config = PredenergyUnifiedConfig(**kwargs)
        
        # Initialize model components
        self.model = None
        self.device = self._setup_device()
        self.is_fitted = False
        
        # Data loaders
        self.data_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        
        print(f"Predenergy model initialized:")
        print(f"  - Model type: {'Combined (STDM + MoTSE)' if self.config.use_combined_model else 'Standard (STDM)'}")
        print(f"  - Sequence length: {self.config.seq_len}")
        print(f"  - Horizon: {self.config.horizon}")
        print(f"  - Device: {self.device}")

    def _setup_device(self) -> str:
        """Setup device for training/inference"""
        if self.config.device == "auto":
            device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
        else:
            device = self.config.device
        
        if device == "gpu" and not paddle.device.is_compiled_with_cuda():
            warnings.warn("GPU requested but not available, falling back to CPU")
            device = "cpu"
        
        return device

    @property
    def model_name(self) -> str:
        """Return model name"""
        return "Predenergy"

    @staticmethod
    def required_hyper_params() -> Dict[str, str]:
        """Return required hyperparameters mapping"""
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "normalize": "scaler"
        }

    def setup_data_loader(
        self, 
        data: Union[str, np.ndarray, pd.DataFrame], 
        **kwargs
    ) -> None:
        """
        Setup data loader for training/inference.
        
        Args:
            data: Training data (file path, array, or DataFrame)
            **kwargs: Additional data loader parameters
        """
        print("Setting up data loader...")
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Create data loader
        self.data_loader = create_Predenergy_data_loader(
            data=data,
            seq_len=self.config.seq_len,
            pred_len=self.config.horizon,
            batch_size=self.config.batch_size,
            features=self.config.features,
            target=self.config.target,
            normalize=self.config.normalize,
            train_ratio=kwargs.get('train_ratio', 0.7),
            val_ratio=kwargs.get('val_ratio', 0.2),
            shuffle=kwargs.get('shuffle', True),
            num_workers=kwargs.get('num_workers', 0),
            loader_type='standard'
        )
        
        # Get individual loaders
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()
        
        print(f"Data loader setup complete:")
        print(f"  - Train batches: {len(self.train_loader)}")
        print(f"  - Validation batches: {len(self.val_loader)}")
        print(f"  - Test batches: {len(self.test_loader)}")

    def _create_model(self) -> None:
        """Create the model instance"""
        if self.model is None:
            print("Creating model...")
            self.model = PredenergyForPrediction(self.config)
            self.model = self.model.to(self.device)
            
            # Setup training components
            self._setup_training_components()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model created:")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")

    def _setup_training_components(self) -> None:
        """Setup optimizer, scheduler, and other training components"""
        if self.model is None:
            return
        
        # Optimizer
        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            parameters=self.model.parameters()
        )
        
        # Learning rate scheduler
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=self.config.learning_rate,
            mode='min',
            factor=0.5,
            patience=self.config.patience // 2,
            verbose=True
        )
        
        # Mixed precision scaler (PaddlePaddle handles this automatically)
        if self.config.mixed_precision and self.device == "gpu":
            paddle.set_amp_level('O1')

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[Dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "Predenergy":
        """
        Fit the model on training data.
        
        Args:
            train_valid_data: Training and validation data
            covariates: Additional covariates (not used in current implementation)
            train_ratio_in_tv: Ratio of training data in train-validation split
            **kwargs: Additional training parameters
            
        Returns:
            Fitted model
        """
        print("Starting model training...")
        
        # Setup data loader if not already done
        if self.data_loader is None:
            self.setup_data_loader(
                data=train_valid_data,
                train_ratio=train_ratio_in_tv,
                **kwargs
            )
        
        # Create model if not already done
        self._create_model()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.is_fitted = True
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return self

    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_data, labels = self._process_batch(batch)
            
            # Forward pass with mixed precision if enabled
            if self.config.mixed_precision and self.device == "gpu":
                with paddle.amp.auto_cast():
                    outputs = self.model(
                        input_data=input_data,
                        labels=labels,
                        return_dict=True
                    )
                    loss = outputs['loss']
                
                # Backward pass
                scaled_loss = paddle.amp.scale_loss(loss, self.optimizer)
                scaled_loss.backward()
                paddle.nn.ClipGradByGlobalNorm(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.clear_grad()
            else:
                outputs = self.model(
                    input_data=input_data,
                    labels=labels,
                    return_dict=True
                )
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                paddle.nn.ClipGradByGlobalNorm(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.clear_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with paddle.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_data, labels = self._process_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    input_data=input_data,
                    labels=labels,
                    return_dict=True
                )
                
                loss = outputs['loss']
                total_loss += loss.numpy().item()
                num_batches += 1
        
        return total_loss / num_batches

    def _process_batch(self, batch):
        """Process batch and move to device"""
        if isinstance(batch, dict):
            input_data = paddle.to_tensor(batch['input_ids'], dtype='float32')
            labels = paddle.to_tensor(batch.get('labels', batch.get('target')), dtype='float32')
        else:
            # Handle tuple format
            input_data = paddle.to_tensor(batch[0], dtype='float32')
            labels = paddle.to_tensor(batch[1], dtype='float32')
        
        return input_data, labels

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Make forecasts.
        
        Args:
            horizon: Forecast horizon
            series: Input time series data
            covariates: Additional covariates (not used)
            
        Returns:
            Forecast predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making forecasts")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        # Convert series to tensor
        if isinstance(series, pd.DataFrame):
            # Take the last seq_len points
            data = series.tail(self.config.seq_len).values
        else:
            data = series
        
        # Ensure correct shape: [1, seq_len, input_size]
        if data.ndim == 1:
            data = data.reshape(1, -1, 1)
        elif data.ndim == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        
        input_tensor = paddle.to_tensor(data, dtype='float32')
        
        # Make prediction
        with paddle.no_grad():
            predictions = self.model.predict(input_tensor, max_horizon_length=horizon)
        
        # Convert to numpy and reshape
        predictions = predictions.numpy()
        
        # Return with correct shape
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)  # Remove batch dimension
        
        return predictions

    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'is_fitted': self.is_fitted,
        }
        
        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        paddle.save(save_dict, path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = paddle.load(path)
        
        # Load configuration
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = PredenergyUnifiedConfig.from_dict(config_dict)
        
        # Create model
        self._create_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.is_fitted = checkpoint.get('is_fitted', True)
        
        print(f"Model loaded from: {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'model_type': 'Combined' if self.config.use_combined_model else 'Standard',
            'seq_len': self.config.seq_len,
            'horizon': self.config.horizon,
            'is_fitted': self.is_fitted,
            'device': self.device,
            'config': self.config.to_dict()
        }
        
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })
        
        return info

    def __repr__(self) -> str:
        """String representation of the model"""
        return f"Predenergy(seq_len={self.config.seq_len}, horizon={self.config.horizon}, fitted={self.is_fitted})"