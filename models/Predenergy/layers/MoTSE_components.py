import math
import warnings
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache, DynamicCache, StaticCache

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)
    The hidden states go from (batch, num_key_value_heads, seq_len, head_dim)
    to (batch, num_attention_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoTSEInputEmbedding(nn.Module):
    """Input embedding layer for MoTSE model."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.motse_hidden_size if hasattr(config, 'motse_hidden_size') else config.hidden_size
        self.input_size = config.motse_input_size if hasattr(config, 'motse_input_size') else config.input_size
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        return self.embedding(x)


class MoTSERotaryEmbedding(torch.nn.Module):
    """Rotary position embedding for MoTSE model."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dim = dim

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MoTSERMSNorm(torch.nn.Module):
    """RMS normalization for MoTSE model."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MoTSETemporalBlock(nn.Module):
    """Temporal block for MoTSE model."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class MoTSEMLP(MoTSETemporalBlock):
    """MLP layer for MoTSE model."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state)


class MoTSESparseExpertsLayer(nn.Module):
    """Sparse experts layer for MoTSE model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok if hasattr(config, 'num_experts_per_tok') else config.motse_num_experts_per_tok
        self.hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.motse_hidden_size
        self.num_experts = config.num_experts if hasattr(config, 'num_experts') else config.motse_num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = self.config.intermediate_size // self.top_k if hasattr(config, 'intermediate_size') else self.config.motse_intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoTSETemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act if hasattr(config, 'hidden_act') else self.config.motse_hidden_act,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = MoTSETemporalBlock(
            hidden_size=self.hidden_size,
            intermediate_size=self.config.intermediate_size if hasattr(config, 'intermediate_size') else self.config.motse_intermediate_size,
            hidden_act=self.config.hidden_act if hasattr(config, 'hidden_act') else self.config.motse_hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MoTSEAttention(nn.Module):
    """Multi-headed attention for MoTSE model."""
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.motse_hidden_size
        self.num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.motse_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.motse_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else config.motse_max_position_embeddings
        self.rope_theta = config.rope_theta if hasattr(config, 'rope_theta') else config.motse_rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout if hasattr(config, 'attention_dropout') else config.motse_attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MoTSERotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MoTSEDecoderLayer(nn.Module):
    """Decoder layer for MoTSE model."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.motse_hidden_size if hasattr(config, 'motse_hidden_size') else config.hidden_size
        self.self_attn = MoTSEAttention(config=config, layer_idx=layer_idx)
        self.mlp = MoTSESparseExpertsLayer(config)
        self.input_layernorm = MoTSERMSNorm(self.hidden_size, eps=config.motse_rms_norm_eps if hasattr(config, 'motse_rms_norm_eps') else config.rms_norm_eps)
        self.post_attention_layernorm = MoTSERMSNorm(self.hidden_size, eps=config.motse_rms_norm_eps if hasattr(config, 'motse_rms_norm_eps') else config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MoTSEOutputLayer(nn.Module):
    """Output layer for MoTSE model."""
    
    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon_length = horizon_length
        self.input_size = input_size
        
        self.output_projection = nn.Linear(hidden_size, horizon_length * input_size)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_dim = x.shape
        
        # Use the last token for prediction
        last_hidden = x[:, -1, :]  # [batch_size, hidden_size]
        
        # Project to output
        output = self.output_projection(last_hidden)  # [batch_size, horizon_length * input_size]
        
        # Reshape to [batch_size, horizon_length, input_size]
        output = output.view(batch_size, self.horizon_length, self.input_size)
        
        return output


# Flash Attention 2 Implementation
class MoTSEFlashAttention2(MoTSEAttention):
    """Flash Attention 2 implementation for MoTSE model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if flash attention is available
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
            self._flash_attn_uses_top_left_mask = True
        except ImportError:
            raise ImportError("Flash Attention 2 is not available. Please install flash-attn package.")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, see: https://github.com/ROCmSoftwarePlatform/flash-attention-release/pull/1/files#diff-ededdbad13610f6f566d41432d9a6adf023531c5d70a6b4fdd709981a1a6718R627
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                indices_q,
                cu_seqlens,
                max_seqlen,
            ) = self._get_unpad_data(attention_mask)

            query_states, key_states, value_states, indices_q, cu_seqlens, max_seqlen = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seqlen

            if not self._flash_attn_uses_top_left_mask:
                causal = self.is_causal and max_seqlen_in_batch_q != 1
            else:
                causal = self.is_causal and max_seqlen_in_batch_q != 1

            if softmax_scale is None:
                softmax_scale = 1 / math.sqrt(self.head_dim)

            attn_output_unpad = flash_attn_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_in_batch_q,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if softmax_scale is None:
                softmax_scale = 1 / math.sqrt(self.head_dim)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        """
        Unpads the input to the Flash Attention implementation.
        """
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
        
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first forward pass when not using the cache, the key and value layers are padded
        # as the sequence length is a multiple of the block size. This creates the incorrect
        # attention scores, as the query layer is not padded. To fix this, we unpad the key and value layers
        # when the query layer is not padded.
        if kv_seq_len != query_length:
            # Assume a left-padded input
            key_layer = index_first_axis(key_layer, attention_mask)
            value_layer = index_first_axis(value_layer, attention_mask)

        if query_length != kv_seq_len:
            # Assume a left-padded input
            query_layer = index_first_axis(query_layer, attention_mask)

        return query_layer, key_layer, value_layer, attention_mask, query_length


# Loss Functions
def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the load balancing loss for MoE models.
    
    Args:
        gate_logits: Logits from the gating mechanism
        top_k: Number of experts to route to
        num_experts: Total number of experts
        attention_mask: Attention mask for padding tokens
        
    Returns:
        Load balancing loss
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts


def _get_unpad_data(attention_mask):
    """
    Get unpad data for Flash Attention.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Weight Initialization
def init_motse_weights(module, config):
    """
    Initialize weights for MoTSE model components.
    
    Args:
        module: The module to initialize
        config: Configuration object
    """
    std = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


# Loss Calculation for Time Series Prediction
def calc_ar_loss(predictions, labels, loss_masks, horizon_length, loss_function=nn.MSELoss()):
    """
    Calculate autoregressive loss for time series prediction.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        loss_masks: Loss masks for padding
        horizon_length: Prediction horizon length
        loss_function: Loss function to use
        
    Returns:
        Calculated loss
    """
    if len(labels.shape) == 2:
        labels.unsqueeze_(dim=-1)
        # enable model parallelism
        labels = labels.to(predictions.device)
    if loss_masks is not None and len(loss_masks.shape) == 2:
        loss_masks.unsqueeze_(dim=-1)
        # enable model parallelism
        loss_masks = loss_masks.to(predictions.device)

    if horizon_length > 1:
        batch_size, seq_len, output_size = predictions.shape
        shift_predictions = predictions.view(batch_size, seq_len, horizon_length, -1)

        # pad to the same length with predictions
        # shape -> [B, input_size, seq_len + horizon_length -1]
        labels = F.pad(labels.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

        # shape -> [B, input_size, seq_len, horizon_length]
        shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
        shift_labels = shift_labels.permute(0, 2, 3, 1)

        if loss_masks is not None:
            # pad to the same length with predictions
            loss_masks = F.pad(loss_masks.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)

            loss_masks = loss_masks.unfold(dimension=-1, size=horizon_length, step=1)
            loss_masks = loss_masks.permute(0, 2, 3, 1)

    else:
        shift_predictions = predictions
        shift_labels = labels

    # Calculate loss with mask
    losses = loss_function(shift_predictions, shift_labels)

    if loss_masks is not None:
        losses = losses * loss_masks
        loss = losses.sum() / loss_masks.sum()
    else:
        loss = torch.mean(losses)

    return loss