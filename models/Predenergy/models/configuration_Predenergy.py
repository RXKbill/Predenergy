from typing import List
from transformers import PretrainedConfig

class PredenergyConfig(PretrainedConfig):
    model_type = "stdm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 512,
            intermediate_size: int = 2048,
            horizon_lengths: List[int] = 1,
            num_hidden_layers: int = 6,
            num_attention_heads: int = 8,
            num_key_value_heads: int = None,
            hidden_act: str = "gelu",
            max_position_embeddings: int = 2048,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-6,
            use_cache: bool = True,
            attention_dropout: float = 0.1,
            hidden_dropout: float = 0.1,
            tie_word_embeddings: bool = False,
            # STDM specific parameters
            seq_len: int = 96,
            pred_len: int = 24,
            label_len: int = 48,
            c_out: int = 1,
            d_model: int = 512,
            n_heads: int = 8,
            e_layers: int = 2,
            d_layers: int = 1,
            d_ff: int = 2048,
            moving_avg: int = 25,
            factor: int = 5,
            distil: bool = True,
            dropout: float = 0.1,
            embed: str = 'timeF',
            freq: str = 'h',
            activation: str = 'gelu',
            output_attention: bool = False,
            **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        # STDM specific parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        ) 