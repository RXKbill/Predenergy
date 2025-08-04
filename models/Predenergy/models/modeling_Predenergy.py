"""
Predenergy Model Implementation
This module provides a corrected implementation of the Predenergy model with proper data flow and dimension handling.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from typing import Dict, Any, Optional, Tuple, Union
import warnings

from models.unified_config import PredenergyUnifiedConfig
from Predenergy.layers.linear_extractor_cluster import Linear_extractor_cluster
from Predenergy.layers.MoTSE_components import (
    MoTSEInputEmbedding, MoTSEDecoderLayer, MoTSEOutputLayer, 
    MoTSERMSNorm, load_balancing_loss_func
)
from Predenergy.utils.masked_attention import (
    Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
)
from layers.configuration_MoTSE import MoTSEConfig
from layers.modeling_MoTSE import MoTSEModel
from models.ts_generation_mixin import TSGenerationMixin


class PredenergyAdaptiveConnection(nn.Module):
    """
    adaptive connection layer between STDM and MoTSE components.
    Handles dimension mismatches and provides multiple connection strategies.
    """
    
    def __init__(self, stdm_output_dim: int, motse_input_dim: int, config: PredenergyUnifiedConfig):
        super(PredenergyAdaptiveConnection, self).__init__()
        self.config = config
        self.stdm_output_dim = stdm_output_dim
        self.motse_input_dim = motse_input_dim
        self.connection_type = config.connection_type
        
        # Different connection strategies
        if self.connection_type == "linear":
            self.projection = nn.Linear(stdm_output_dim, motse_input_dim)
        
        elif self.connection_type == "attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=motse_input_dim,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True
            )
            self.projection = nn.Linear(stdm_output_dim, motse_input_dim)
            
        elif self.connection_type == "concat":
            # Concatenate and project
            self.projection = nn.Linear(stdm_output_dim + motse_input_dim, motse_input_dim)
            
        elif self.connection_type == "adaptive":
            # Learnable gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(stdm_output_dim, motse_input_dim),
                nn.Sigmoid()
            )
            self.projection = nn.Linear(stdm_output_dim, motse_input_dim)
            self.layer_norm = nn.LayerNorm(motse_input_dim)
        
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")
    
    def forward(self, stdm_output: paddle.Tensor, motse_input: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """
        Forward pass of the adaptive connection layer.
        
        Args:
            stdm_output: Output from STDM component [batch_size, seq_len, stdm_dim]
            motse_input: Optional input for MoTSE (for certain connection types)
        
        Returns:
            Connected features ready for MoTSE [batch_size, seq_len, motse_dim]
        """
        if self.connection_type == "linear":
            output = self.projection(stdm_output)
            
        elif self.connection_type == "attention":
            # Project STDM output to MoTSE dimension
            projected = self.projection(stdm_output)
            
            if motse_input is not None:
                # Use cross-attention
                output, _ = self.cross_attention(projected, motse_input, motse_input)
            else:
                # Self-attention on projected features
                output, _ = self.cross_attention(projected, projected, projected)
                
        elif self.connection_type == "concat":
            if motse_input is None:
                # Create dummy input with same dimensions as STDM output
                motse_input = paddle.zeros_like(stdm_output)
                
            # Concatenate along feature dimension
            concatenated = paddle.concat([stdm_output, motse_input], axis=-1)
            output = self.projection(concatenated)
            
        elif self.connection_type == "adaptive":
            projected = self.projection(stdm_output)
            gate_weights = self.gate(stdm_output)
            
            if motse_input is not None:
                # Adaptive fusion
                output = gate_weights * projected + (1 - gate_weights) * motse_input
            else:
                output = projected
                
            output = self.layer_norm(output)
        
        return output


class PredenergyModel(nn.Module):
    """
     Predenergy model with proper data flow and dimension handling.
    This version resolves the issues in the original implementation.
    """
    
    def __init__(self, config: PredenergyUnifiedConfig):
        super(PredenergyModel, self).__init__()
        self.config = config
        
        # Input embedding layer
        self.input_embedding = nn.Linear(config.input_size, config.d_model)
        
        # STDM components
        self.cluster = Linear_extractor_cluster(self._get_cluster_config(config))
        self.CI = config.CI
        self.n_vars = getattr(config, 'enc_in', config.input_size)
        
        # Attention mask generator (only for multivariate)
        if self.n_vars > 1:
            self.mask_generator = Mahalanobis_mask(config.seq_len)
            self.channel_transformer = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                config.factor,
                                attention_dropout=config.dropout,
                                output_attention=config.output_attention,
                            ),
                            config.d_model,
                            config.n_heads,
                        ),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                    for _ in range(config.e_layers)
                ],
                norm_layer=nn.LayerNorm(config.d_model)
            )
        
        # Combined model components
        if config.use_combined_model:
            self.connection = PredenergyAdaptiveConnection(
                stdm_output_dim=config.d_model,
                motse_input_dim=config.motse_hidden_size,
                config=config
            )
            
            # MoTSE configuration
            motse_config = self._create_motse_config(config)
            self.motse_model = MoTSEModel(motse_config)
            self.motse_output_layer = MoTSEOutputLayer(
                hidden_size=config.motse_hidden_size,
                horizon_length=config.horizon,
                input_size=config.input_size
            )
            
            # Final output projection
            self.final_output = nn.Linear(config.motse_hidden_size, config.horizon * config.c_out)
        else:
            # Standard model final projection
            self.final_output = nn.Linear(config.d_model, config.horizon * config.c_out)
        
        # Optional layer normalization
        if config.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.horizon * config.c_out)
        
        self._init_weights()
    
    def _get_cluster_config(self, config: PredenergyUnifiedConfig) -> Dict[str, Any]:
        """Create configuration for the cluster component."""
        return {
            'seq_len': config.seq_len,
            'd_model': config.d_model,
            'enc_in': getattr(config, 'enc_in', config.input_size),
            'dropout': config.dropout,
            'factor': config.factor,
            'num_experts': config.num_experts if config.use_combined_model else 1,
            'num_experts_per_tok': config.num_experts_per_tok if config.use_combined_model else 1,
        }
    
    def _create_motse_config(self, config: PredenergyUnifiedConfig) -> MoTSEConfig:
        """Create MoTSE configuration from unified config."""
        return MoTSEConfig(
            input_size=config.input_size,
            hidden_size=config.motse_hidden_size,
            intermediate_size=config.motse_intermediate_size,
            num_hidden_layers=config.motse_num_layers,
            num_attention_heads=config.motse_num_heads,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            horizon_lengths=[config.horizon],
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            router_aux_loss_factor=config.router_aux_loss_factor,
            apply_aux_loss=config.apply_aux_loss,
            use_cache=config.use_cache,
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_data: paddle.Tensor, return_attention: bool = False) -> Dict[str, paddle.Tensor]:
        """
        Forward pass of the Predenergy model.
        
        Args:
            input_data: Input tensor [batch_size, seq_len, input_size]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        batch_size, seq_len, input_size = input_data.shape
        
        # Validate input dimensions
        if seq_len != self.config.seq_len:
            warnings.warn(f"Input sequence length {seq_len} doesn't match config {self.config.seq_len}")
        
        # Input embedding
        embedded_input = self.input_embedding(input_data)
        
        # STDM forward pass
        if self.CI:
            # Channel-independent processing
            channel_independent_input = rearrange(embedded_input, 'b l d -> (b d) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b d) l 1 -> b l d', b=batch_size)
        else:
            temporal_feature, L_importance = self.cluster(embedded_input)
        
        # Channel-wise processing for multivariate data
        if self.n_vars > 1:
            # Reshape for channel transformer: [batch_size, n_vars, d_model]
            temporal_feature = rearrange(temporal_feature, 'b l d -> b d l')
            
            # Generate attention mask
            channel_mask = self.mask_generator(temporal_feature)
            
            # Apply channel transformer
            channel_feature, attention = self.channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )
            stdm_output = channel_feature
        else:
            stdm_output = temporal_feature
            attention = None
        
        # Final processing
        if self.config.use_combined_model:
            # Combined model: STDM + MoTSE
            connected_features = self.connection(stdm_output)
            
            # MoTSE forward pass
            motse_outputs = self.motse_model(
                input_ids=connected_features,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=return_attention,
                output_hidden_states=False,
                return_dict=True
            )
            
            # Get MoTSE hidden states and router logits
            motse_hidden_states = motse_outputs.last_hidden_state
            router_logits = getattr(motse_outputs, 'router_logits', None)
            
            # MoTSE output layer
            motse_predictions = self.motse_output_layer(motse_hidden_states)
            
            # Final output projection
            final_hidden = motse_hidden_states.mean(axis=1)  # Global average pooling
            final_output = self.final_output(final_hidden)
            
        else:
            # Standard model: STDM only
            final_hidden = stdm_output.mean(axis=1)  # Global average pooling
            final_output = self.final_output(final_hidden)
            motse_predictions = None
            router_logits = None
        
        # Apply layer normalization if enabled
        if hasattr(self, 'final_norm'):
            final_output = self.final_norm(final_output)
        
        # Reshape output: [batch_size, horizon, c_out]
        final_output = final_output.reshape([batch_size, self.config.horizon, self.config.c_out])
        
        # Prepare return dictionary
        outputs = {
            'predictions': final_output,
            'L_importance': L_importance,
        }
        
        if self.config.use_combined_model:
            outputs.update({
                'motse_predictions': motse_predictions,
                'router_logits': router_logits,
            })
        
        if return_attention:
            outputs['attention_weights'] = attention
            if self.config.use_combined_model and hasattr(motse_outputs, 'attentions'):
                outputs['motse_attention_weights'] = motse_outputs.attentions
        
        return outputs


class PredenergyForPrediction(nn.Module, TSGenerationMixin):
    """
    Predenergy model for prediction tasks with proper loss calculation and generation support.
    """
    
    def __init__(self, config: PredenergyUnifiedConfig):
        super(PredenergyForPrediction, self).__init__()
        self.config = config
        self.model = PredenergyModel(config)
        
        # Loss function
        if config.loss_function == "huber":
            self.loss_function = nn.HuberLoss(reduction='none', delta=2.0)
        elif config.loss_function == "mse":
            self.loss_function = nn.MSELoss(reduction='none')
        elif config.loss_function == "mae":
            self.loss_function = nn.L1Loss(reduction='none')
        else:
            self.loss_function = nn.HuberLoss(reduction='none', delta=2.0)
        
        # Multi-scale prediction heads (if needed)
        if hasattr(config, 'horizon_lengths') and len(config.horizon_lengths) > 1:
            self.multi_scale_heads = nn.ModuleList([
                nn.Linear(config.motse_hidden_size if config.use_combined_model else config.d_model, 
                         horizon_length * config.c_out)
                for horizon_length in config.horizon_lengths
            ])
            
            self.horizon_length_map = {
                horizon_length: i for i, horizon_length in enumerate(config.horizon_lengths)
            }
        else:
            self.multi_scale_heads = None
            self.horizon_length_map = {config.horizon: 0}
    
    def forward(
        self, 
        input_data: paddle.Tensor, 
        labels: Optional[paddle.Tensor] = None, 
        loss_masks: Optional[paddle.Tensor] = None,
        max_horizon_length: Optional[int] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Dict[str, paddle.Tensor], Tuple]:
        """
        Forward pass with optional loss calculation.
        
        Args:
            input_data: Input tensor [batch_size, seq_len, input_size]
            labels: Target labels [batch_size, horizon, c_out]
            loss_masks: Loss masks [batch_size, horizon]
            max_horizon_length: Maximum horizon length for generation
            use_cache: Whether to use cache (for generation)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dictionary
        
        Returns:
            Model outputs including predictions and loss
        """
        # Get model outputs
        outputs = self.model(input_data, return_attention=output_attentions)
        
        predictions = outputs['predictions']
        loss = None
        aux_loss = None
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss = self._calc_prediction_loss(predictions, labels, loss_masks)
            
            # Calculate auxiliary loss for combined model
            if self.config.use_combined_model and outputs.get('router_logits') is not None:
                aux_loss = self._calc_load_balancing_loss(outputs['router_logits'])
                loss = loss + aux_loss
        
        # Prepare return values
        result = {
            'predictions': predictions,
            'loss': loss,
            'aux_loss': aux_loss,
            **outputs
        }
        
        if output_hidden_states:
            result['hidden_states'] = outputs.get('motse_hidden_states')
        
        if return_dict:
            return result
        else:
            return (predictions, loss)
    
    def _calc_prediction_loss(
        self, 
        predictions: paddle.Tensor, 
        labels: paddle.Tensor, 
        loss_masks: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Calculate prediction loss."""
        # Ensure dimensions match
        if predictions.shape != labels.shape:
            raise ValueError(f"Prediction shape {predictions.shape} doesn't match label shape {labels.shape}")
        
        # Calculate element-wise loss
        element_loss = self.loss_function(predictions, labels)
        
        # Apply loss masks if provided
        if loss_masks is not None:
            if loss_masks.ndim == 2:  # [batch_size, horizon]
                loss_masks = loss_masks.unsqueeze(-1)  # [batch_size, horizon, 1]
            element_loss = element_loss * loss_masks
        
        # Return mean loss
        return element_loss.mean()
    
    def _calc_load_balancing_loss(
        self, 
        router_logits: paddle.Tensor, 
        attention_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Calculate load balancing loss for MoE."""
        if router_logits is None:
            return paddle.to_tensor(0.0, dtype='float32')
        
        return load_balancing_loss_func(
            router_logits,
            top_k=self.config.num_experts_per_tok,
            num_experts=self.config.num_experts,
            attention_mask=attention_mask
        ) * self.config.router_aux_loss_factor
    
    def predict(self, input_data: paddle.Tensor, max_horizon_length: Optional[int] = None) -> paddle.Tensor:
        """Make predictions without loss calculation."""
        with paddle.no_grad():
            outputs = self.forward(
                input_data, 
                max_horizon_length=max_horizon_length,
                return_dict=True
            )
            return outputs['predictions']
    
    def generate(self, input_data: paddle.Tensor, max_length: Optional[int] = None, **kwargs) -> paddle.Tensor:
        """Generate predictions using autoregressive approach (if needed)."""
        # For now, just return regular prediction
        # This can be extended for more sophisticated generation strategies
        return self.predict(input_data, max_horizon_length=max_length)