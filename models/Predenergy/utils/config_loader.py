"""
Updated Configuration loader for Predenergy models
This module provides utilities for loading and managing the unified model configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path

from models.unified_config import PredenergyUnifiedConfig, load_config_from_file, save_config_to_file


class ConfigLoader:
    """
    Updated Configuration loader for Predenergy models using unified config system.
    This class provides backward compatibility while using the new unified configuration.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.default_configs = {
            'standard': 'predenergy_config.yaml',
            'combined': 'predenergy_stdm_motse.yaml'
        }
    
    def load_config(self, config_path: Optional[str] = None, 
                   config_type: str = 'standard') -> PredenergyUnifiedConfig:
        """
        Load configuration from file or use default
        
        Args:
            config_path: Path to configuration file
            config_type: Type of configuration ('standard' or 'combined')
            
        Returns:
            PredenergyUnifiedConfig object
        """
        if config_path:
            return load_config_from_file(config_path)
        else:
            return self._load_default_config(config_type)
    
    def _load_default_config(self, config_type: str) -> PredenergyUnifiedConfig:
        """Load default configuration"""
        if config_type not in self.default_configs:
            raise ValueError(f"Unknown config type: {config_type}. Available types: {list(self.default_configs.keys())}")
        
        config_file = self.config_dir / self.default_configs[config_type]
        
        # If default config file doesn't exist, create a basic one
        if not config_file.exists():
            print(f"Default config file {config_file} not found, creating basic configuration")
            return self._create_basic_config(config_type)
        
        return load_config_from_file(str(config_file))
    
    def _create_basic_config(self, config_type: str) -> PredenergyUnifiedConfig:
        """Create a basic configuration when default file is missing"""
        basic_config = PredenergyUnifiedConfig()
        if config_type == 'combined':
            basic_config.use_combined_model = True
        return basic_config
    
    def create_model_config(self, config_dict: Dict[str, Any], 
                          use_combined_model: bool = False) -> PredenergyUnifiedConfig:
        """
        Create model configuration object from dictionary
        
        Args:
            config_dict: Configuration dictionary
            use_combined_model: Whether to use combined model
            
        Returns:
            PredenergyUnifiedConfig object
        """
        # Clean and validate configuration dictionary
        cleaned_config = self._clean_config_dict(config_dict)
        
        # Set combined model flag
        cleaned_config['use_combined_model'] = use_combined_model
        
        # Create configuration object
        config = PredenergyUnifiedConfig.from_dict(cleaned_config)
        
        return config
    
    def _clean_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate configuration dictionary"""
        # Remove non-configuration parameters
        non_config_keys = {
            'data_path', 'output_dir', 'model_name', 'save_config', 
            'seed', 'config', 'model_path', 'test_data_path'
        }
        cleaned_config = {k: v for k, v in config_dict.items() if k not in non_config_keys}
        
        # Handle legacy parameter names
        legacy_mapping = {
            'pred_len': 'horizon',
            'enc_in': 'input_size',
            'dec_in': 'input_size',
            'num_workers': None,  # Remove this parameter
            'shuffle': None,      # Remove this parameter
        }
        
        for old_key, new_key in legacy_mapping.items():
            if old_key in cleaned_config:
                if new_key is not None:
                    cleaned_config[new_key] = cleaned_config[old_key]
                del cleaned_config[old_key]
        
        return cleaned_config
    
    def validate_config(self, config: PredenergyUnifiedConfig) -> bool:
        """
        Validate configuration object
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # The validation is handled in the config's __post_init__ method
            # Just check if it's a valid config object
            return isinstance(config, PredenergyUnifiedConfig)
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def save_config(self, config: PredenergyUnifiedConfig, output_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        save_config_to_file(config, output_path)
        print(f"Configuration saved to {output_path}")
    
    def get_config_summary(self, config: PredenergyUnifiedConfig) -> str:
        """
        Get a human-readable summary of the configuration
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Summary string
        """
        summary = []
        summary.append("=== Predenergy Configuration Summary ===")
        summary.append(f"Model Type: {'Combined (STDM + MoTSE)' if config.use_combined_model else 'Standard (STDM)'}")
        summary.append(f"Sequence Length: {config.seq_len}")
        summary.append(f"Prediction Horizon: {config.horizon}")
        summary.append(f"Model Dimension: {config.d_model}")
        summary.append(f"Attention Heads: {config.n_heads}")
        summary.append(f"Encoder Layers: {config.e_layers}")
        summary.append(f"Decoder Layers: {config.d_layers}")
        summary.append(f"Features: {config.features}")
        summary.append(f"Batch Size: {config.batch_size}")
        summary.append(f"Learning Rate: {config.learning_rate}")
        
        if config.use_combined_model:
            summary.append("--- MoTSE Parameters ---")
            summary.append(f"Experts: {config.num_experts}")
            summary.append(f"Experts per Token: {config.num_experts_per_tok}")
            summary.append(f"Connection Type: {config.connection_type}")
            summary.append(f"MoTSE Hidden Size: {config.motse_hidden_size}")
        
        summary.append(f"Device: {config.device}")
        summary.append("=" * 40)
        
        return "\n".join(summary)


def load_predenergy_config(config_path: Optional[str] = None, 
                          use_combined_model: bool = False) -> PredenergyUnifiedConfig:
    """
    Convenience function to load Predenergy configuration
    
    Args:
        config_path: Path to configuration file
        use_combined_model: Whether to use combined model
        
    Returns:
        PredenergyUnifiedConfig object
    """
    loader = ConfigLoader()
    
    if config_path:
        config = loader.load_config(config_path)
    else:
        config_type = 'combined' if use_combined_model else 'standard'
        config = loader.load_config(config_type=config_type)
    
    # Override the combined model setting if explicitly specified
    if use_combined_model:
        config.use_combined_model = True
    
    return config


def create_config_from_args(args: Dict[str, Any]) -> PredenergyUnifiedConfig:
    """
    Create configuration from command line arguments or similar dictionary
    
    Args:
        args: Dictionary of arguments
        
    Returns:
        PredenergyUnifiedConfig object
    """
    loader = ConfigLoader()
    config = loader.create_model_config(
        args, 
        use_combined_model=args.get('use_combined_model', False)
    )
    return config