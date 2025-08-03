#!/usr/bin/env python
# -*- coding:utf-8 _*-
import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_Predenergy import PredenergyConfig
from ..layers.Embed import DataEmbedding
from ..layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
from ..layers.SelfAttention_Family import ProbAttention, FullAttention
from ..layers.Transformer_EncDec import ConvLayer
from ..layers.linear_pattern_extractor import Linear_extractor
from .ts_generation_mixin import TSGenerationMixin


class PredenergyPreTrainedModel(PreTrainedModel):
    config_class = PredenergyConfig
    base_model_prefix = "Predenergy"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PredenergyLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PredenergyModel(PredenergyPreTrainedModel):
    def __init__(self, config: PredenergyConfig):
        super().__init__(config)
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            config.input_size, config.d_model, config.embed, config.freq, config.dropout
        )
        self.dec_embedding = DataEmbedding(
            config.c_out, config.d_model, config.embed, config.freq, config.dropout
        )

        # Attention
        Attention = ProbAttention if config.factor == 5 else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attention(False, config.factor, attention_dropout=config.dropout, output_attention=config.output_attention),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    Attention(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                    Attention(False, config.factor, attention_dropout=config.dropout, output_attention=config.output_attention),
                    config.d_model,
                    config.c_out,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                ) for _ in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True)
        )

        # Linear pattern extractor
        self.linear_extractor = Linear_extractor(config, individual=False)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # Linear pattern extraction
        linear_out = self.linear_extractor(x_enc)

        # Combine predictions
        if self.config.output_attention:
            return dec_out, attns, linear_out
        else:
            return dec_out, linear_out


class PredenergyForPrediction(PredenergyPreTrainedModel, TSGenerationMixin):
    def __init__(self, config: PredenergyConfig):
        super().__init__(config)
        self.config = config
        self.Predenergymodel = PredenergyModel(config)
        
        # Prediction head
        self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
    ):
        
        # Get model outputs
        if self.config.output_attention:
            dec_out, attns, linear_out = self.Predenergymodel(
                x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask, dec_self_mask, dec_enc_mask
            )
        else:
            dec_out, linear_out = self.Predenergymodel(
                x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask, dec_self_mask, dec_enc_mask
            )

        # Project to output dimension
        dec_out = self.projection(dec_out)
        
        # Combine with linear pattern
        outputs = dec_out + linear_out

        loss = None
        if labels is not None:
            # Calculate loss
            if loss_masks is not None:
                loss = nn.MSELoss()(outputs * loss_masks, labels * loss_masks)
            else:
                loss = nn.MSELoss()(outputs, labels)

        return {
            'loss': loss,
            'predictions': outputs,
            'attention': attns if self.config.output_attention else None
        }

    def predict(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, 
                x_dec: torch.Tensor, x_mark_dec: torch.Tensor):
        """Prediction method for inference"""
        with torch.no_grad():
            outputs = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return outputs['predictions']

    def generate(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, 
                max_length: Optional[int] = None, **kwargs):
        """Generate predictions using the generation mixin"""
        # This would be implemented based on the specific generation requirements
        # For now, return the prediction method
        return self.predict(x_enc, x_mark_enc, x_enc, x_mark_enc) 