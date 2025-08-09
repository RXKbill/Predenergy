"""
Unified Configuration System for Predenergy Models
This module provides a single, consistent configuration system for all Predenergy model variants.
"""

import warnings
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import paddle


@dataclass
class PredenergyUnifiedConfig:
    """
    Unified configuration class for all Predenergy model variants.
    This replaces the multiple conflicting configuration classes.
    """
    
    # ===== Core Prediction Parameters =====
    seq_len: int = 96                    # Input sequence length
    horizon: int = 24                    # Prediction horizon
    label_len: int = 48                  # Label length for decoder
    input_size: int = 1                  # Input feature dimension
    c_out: int = 1                       # Output dimension
    
    # ===== Model Architecture =====
    d_model: int = 512                   # Model dimension
    n_heads: int = 8                     # Number of attention heads
    e_layers: int = 2                    # Number of encoder layers
    d_layers: int = 1                    # Number of decoder layers
    d_ff: int = 2048                     # Feedforward dimension
    dropout: float = 0.1                 # Dropout rate
    activation: str = "gelu"             # Activation function
    
    # ===== Attention Parameters =====
    factor: int = 5                      # Attention factor
    output_attention: bool = False       # Output attention weights
    
    # ===== Training Parameters =====
    batch_size: int = 32                 # Batch size
    learning_rate: float = 0.001         # Learning rate
    num_epochs: int = 100                # Number of epochs
    patience: int = 10                   # Early stopping patience
    loss_function: str = "huber"         # Loss function type
    
    # ===== Data Processing =====
    features: str = "S"                  # S for univariate, M for multivariate
    target: str = "OT"                   # Target column name
    normalize: int = 2                   # Normalization method
    freq: str = "h"                      # Frequency string
    embed: str = "timeF"                 # Embedding type
    
    # ===== Advanced Model Options =====
    use_layer_norm: bool = True          # Use layer normalization
    use_revin: bool = True               # Use RevIN normalization
    moving_avg: int = 25                 # Moving average window
    
    # ===== STDM Specific Parameters =====
    CI: bool = True                      # Channel independence
    distil: bool = True                  # Use distillation
    
    # ===== MoTSE Architecture Parameters =====
    num_experts: int = 8                 # Number of experts
    num_experts_per_tok: int = 2         # Experts per token
    connection_type: str = "adaptive"    # Connection type: linear, attention, concat, adaptive
    motse_hidden_size: int = 1024        # MoTSE hidden size
    motse_num_layers: int = 6            # MoTSE number of layers
    motse_num_heads: int = 16            # MoTSE attention heads
    motse_intermediate_size: int = 4096  # MoTSE intermediate size
    router_aux_loss_factor: float = 0.02 # Router auxiliary loss factor
    apply_aux_loss: bool = True          # Apply auxiliary loss
    
    # ===== PaddleNLP Decoder Parameters =====
    use_paddlenlp_decoder: bool = True   # Use PaddleNLP decoder for feature decoding
    decoder_hidden_size: int = 512       # Decoder hidden size
    decoder_num_layers: int = 3          # Number of decoder layers
    decoder_num_heads: int = 8           # Number of attention heads in decoder
    decoder_dropout: float = 0.1         # Decoder dropout rate
    
    # ===== Generation Parameters =====
    max_position_embeddings: int = 2048  # Maximum position embeddings
    rope_theta: float = 10000.0          # RoPE theta parameter
    use_cache: bool = True               # Use KV cache
    
    # ===== Device and Performance =====
    device: str = "auto"                 # Device selection
    mixed_precision: bool = True         # Use mixed precision training
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        self._validate_config()
        self._adjust_dependent_params()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.features not in ["S", "M"]:
            raise ValueError("features must be 'S' or 'M'")
        if self.connection_type not in ["linear", "attention", "concat", "adaptive"]:
            raise ValueError("connection_type must be one of: linear, attention, concat, adaptive")
    
    def _adjust_dependent_params(self):
        """Adjust dependent parameters based on configuration."""
        # Ensure consistency between horizon and pred_len
        self.pred_len = self.horizon
        
        # Adjust encoder/decoder input dimensions based on features
        if self.features == "S":
            self.enc_in = self.input_size
            self.dec_in = self.input_size
        else:
            # For multivariate, these should be set based on actual data
            if not hasattr(self, 'enc_in'):
                self.enc_in = self.input_size
            if not hasattr(self, 'dec_in'):
                self.dec_in = self.input_size
        
        # Device configuration
        if self.device == "auto":
            self.device = "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
    
    def get_motse_config(self) -> Dict[str, Any]:
        """Get MoTSE-specific configuration parameters."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.motse_hidden_size,
            "intermediate_size": self.motse_intermediate_size,
            "num_hidden_layers": self.motse_num_layers,
            "num_attention_heads": self.motse_num_heads,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "horizon_lengths": [self.horizon],
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "router_aux_loss_factor": self.router_aux_loss_factor,
            "apply_aux_loss": self.apply_aux_loss,
            "use_cache": self.use_cache,
        }
    
    def get_decoder_config(self) -> Dict[str, Any]:
        """Get PaddleNLP Decoder configuration parameters."""
        return {
            "hidden_size": self.decoder_hidden_size,
            "num_layers": self.decoder_num_layers,
            "num_heads": self.decoder_num_heads,
            "dropout": self.decoder_dropout,
            "vocab_size": self.horizon * self.c_out,  # Output vocabulary size
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": 1e-5,
        }
    
    def get_stdm_config(self) -> Dict[str, Any]:
        """Get STDM-specific configuration parameters."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "enc_in": self.enc_in,
            "dec_in": self.dec_in,
            "c_out": self.c_out,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "activation": self.activation,
            "factor": self.factor,
            "output_attention": self.output_attention,
            "moving_avg": self.moving_avg,
            "CI": self.CI,
            "distil": self.distil,
            "embed": self.embed,
            "freq": self.freq,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration parameters."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "loss_function": self.loss_function,
            "mixed_precision": self.mixed_precision,
            "device": self.device,
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration parameters."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": self.features,
            "target": self.target,
            "normalize": self.normalize,
            "freq": self.freq,
            "batch_size": self.batch_size,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PredenergyUnifiedConfig":
        """Create configuration from dictionary."""
        # Filter out unknown parameters
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        lines = [f"{self.__class__.__name__}("]
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            lines.append(f"    {field.name}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# Compatibility aliases for backward compatibility
PredenergyConfig = PredenergyUnifiedConfig
TransformerConfig = PredenergyUnifiedConfig
PredenergyMoTSEConfig = PredenergyUnifiedConfig


def load_config_from_file(config_path: str) -> PredenergyUnifiedConfig:
    """Load configuration from YAML or JSON file."""
    import yaml
    import json
    from pathlib import Path
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return PredenergyUnifiedConfig.from_dict(config_dict)


def save_config_to_file(config: PredenergyUnifiedConfig, config_path: str) -> None:
    """Save configuration to YAML or JSON file."""
    import yaml
    import json
    from pathlib import Path
    
    config_path = Path(config_path)
    config_dict = config.to_dict()
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")