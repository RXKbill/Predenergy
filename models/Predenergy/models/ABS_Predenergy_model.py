import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Predenergy.layers.linear_extractor_cluster import Linear_extractor_cluster
from Predenergy.layers.MoTSE_components import MoTSEInputEmbedding, MoTSEDecoderLayer, MoTSEOutputLayer, MoTSERMSNorm
from Predenergy.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer

# 导入MoTSE的配置和模型组件
from layers.configuration_MoTSE import MoTSEConfig
from layers.modeling_MoTSE import MoTSEModel, MoTSEForPrediction
from models.ts_generation_mixin import TSGenerationMixin


class PredenergyMoTSEConfig:
    """Configuration class for Predenergy combined model."""
    
    def __init__(self, **kwargs):
        # Predenergy parameters
        self.enc_in = kwargs.get('enc_in', 1)
        self.dec_in = kwargs.get('dec_in', 1)
        self.c_out = kwargs.get('c_out', 1)
        self.e_layers = kwargs.get('e_layers', 2)
        self.d_layers = kwargs.get('d_layers', 1)
        self.d_model = kwargs.get('d_model', 512)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.freq = kwargs.get('freq', 'h')
        self.factor = kwargs.get('factor', 1)
        self.n_heads = kwargs.get('n_heads', 8)
        self.seg_len = kwargs.get('seg_len', 6)
        self.win_size = kwargs.get('win_size', 2)
        self.activation = kwargs.get('activation', 'gelu')
        self.output_attention = kwargs.get('output_attention', 0)
        self.patch_len = kwargs.get('patch_len', 16)
        self.stride = kwargs.get('stride', 8)
        self.period_len = kwargs.get('period_len', 4)
        self.dropout = kwargs.get('dropout', 0.2)
        self.fc_dropout = kwargs.get('fc_dropout', 0.2)
        self.moving_avg = kwargs.get('moving_avg', 25)
        self.batch_size = kwargs.get('batch_size', 256)
        self.lradj = kwargs.get('lradj', 'type3')
        self.lr = kwargs.get('lr', 0.02)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.num_workers = kwargs.get('num_workers', 0)
        self.loss = kwargs.get('loss', 'huber')
        self.patience = kwargs.get('patience', 10)
        self.num_experts = kwargs.get('num_experts', 4)
        self.noisy_gating = kwargs.get('noisy_gating', True)
        self.k = kwargs.get('k', 1)
        self.CI = kwargs.get('CI', True)
        self.seq_len = kwargs.get('seq_len', 96)
        self.horizon = kwargs.get('horizon', 24)
        self.pred_len = kwargs.get('pred_len', 24)
        
        self.motse_config = MoTSEConfig(
            hidden_size=kwargs.get('motse_hidden_size', 512),
            intermediate_size=kwargs.get('motse_intermediate_size', 1024),
            num_hidden_layers=kwargs.get('motse_num_hidden_layers', 4),
            num_attention_heads=kwargs.get('motse_num_attention_heads', 8),
            num_key_value_heads=kwargs.get('motse_num_key_value_heads', 8),
            hidden_act=kwargs.get('motse_hidden_act', 'silu'),
            num_experts_per_tok=kwargs.get('motse_num_experts_per_tok', 2),
            num_experts=kwargs.get('motse_num_experts', 4),
            max_position_embeddings=kwargs.get('motse_max_position_embeddings', 2048),
            initializer_range=kwargs.get('motse_initializer_range', 0.02),
            rms_norm_eps=kwargs.get('motse_rms_norm_eps', 1e-6),
            use_cache=kwargs.get('motse_use_cache', True),
            use_dense=kwargs.get('motse_use_dense', False),
            rope_theta=kwargs.get('motse_rope_theta', 10000),
            attention_dropout=kwargs.get('motse_attention_dropout', 0.0),
            apply_aux_loss=kwargs.get('motse_apply_aux_loss', True),
            router_aux_loss_factor=kwargs.get('motse_router_aux_loss_factor', 0.02),
            tie_word_embeddings=kwargs.get('motse_tie_word_embeddings', False),
            input_size=kwargs.get('motse_input_size', 1),
            horizon_lengths=kwargs.get('motse_horizon_lengths', [24]),
            _attn_implementation=kwargs.get('motse_attn_implementation', 'eager')
        )
        
        # Connection parameters
        self.connection_type = kwargs.get('connection_type', 'adaptive')  # 'linear', 'attention', 'concat', 'adaptive'
        self.connection_dropout = kwargs.get('connection_dropout', 0.1)
        self.connection_hidden_size = kwargs.get('connection_hidden_size', 256)
        
        # Advanced connection parameters
        self.use_cross_attention = kwargs.get('use_cross_attention', True)
        self.use_residual_connection = kwargs.get('use_residual_connection', True)
        self.use_layer_norm = kwargs.get('use_layer_norm', True)
        
        # MoTSE specific parameters for prediction
        self.apply_aux_loss = kwargs.get('apply_aux_loss', True)
        self.router_aux_loss_factor = kwargs.get('router_aux_loss_factor', 0.02)


class PredenergyAdaptiveConnection(nn.Module):
    """自适应连接层，支持多种连接策略"""
    
    def __init__(self, stdm_output_dim, motse_input_dim, config):
        super().__init__()
        self.config = config
        self.connection_type = config.connection_type
        self.dropout = nn.Dropout(config.connection_dropout)
        
        # 基础连接层
        if self.connection_type == 'linear':
            self.connection_layer = nn.Linear(stdm_output_dim, motse_input_dim)
        elif self.connection_type == 'attention':
            self.attention = nn.MultiheadAttention(motse_input_dim, num_heads=8, batch_first=True)
            self.connection_layer = nn.Linear(stdm_output_dim, motse_input_dim)
        elif self.connection_type == 'concat':
            self.connection_layer = nn.Linear(stdm_output_dim + motse_input_dim, motse_input_dim)
        elif self.connection_type == 'adaptive':
            # 自适应连接：结合多种策略
            self.linear_proj = nn.Linear(stdm_output_dim, motse_input_dim)
            self.attention = nn.MultiheadAttention(motse_input_dim, num_heads=8, batch_first=True)
            self.gate = nn.Linear(motse_input_dim * 2, motse_input_dim)
            self.layer_norm = nn.LayerNorm(motse_input_dim) if config.use_layer_norm else None
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")
        
        # 交叉注意力（如果启用）
        if config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(motse_input_dim, num_heads=8, batch_first=True)
            self.cross_norm = nn.LayerNorm(motse_input_dim) if config.use_layer_norm else None
    
    def forward(self, stdm_output, motse_input=None):
        """
        Args:
            stdm_output: [batch_size, seq_len, stdm_output_dim]
            motse_input: [batch_size, seq_len, motse_input_dim] (optional)
        """
        if self.connection_type == 'linear':
            connected = self.connection_layer(stdm_output)
            return self.dropout(connected)
        
        elif self.connection_type == 'attention':
            connected = self.connection_layer(stdm_output)
            connected, _ = self.attention(connected, connected, connected)
            return self.dropout(connected)
        
        elif self.connection_type == 'concat':
            if motse_input is None:
                raise ValueError("motse_input is required for concat connection type")
            concatenated = torch.cat([stdm_output, motse_input], dim=-1)
            connected = self.connection_layer(concatenated)
            return self.dropout(connected)
        
        elif self.connection_type == 'adaptive':
            # 线性投影
            linear_out = self.linear_proj(stdm_output)
            
            # 自注意力处理
            attn_out, _ = self.attention(linear_out, linear_out, linear_out)
            
            # 门控机制
            if motse_input is not None:
                gate_input = torch.cat([attn_out, motse_input], dim=-1)
                gate_output = torch.sigmoid(self.gate(gate_input))
                connected = gate_output * attn_out + (1 - gate_output) * motse_input
            else:
                connected = attn_out
            
            # 交叉注意力（如果启用）
            if self.config.use_cross_attention and motse_input is not None:
                cross_out, _ = self.cross_attention(connected, motse_input, motse_input)
                if self.cross_norm is not None:
                    cross_out = self.cross_norm(cross_out)
                connected = cross_out + connected if self.config.use_residual_connection else cross_out
            
            # 层归一化
            if self.layer_norm is not None:
                connected = self.layer_norm(connected)
            
            return self.dropout(connected)
        
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")


class PredenergyModel(nn.Module):
    """改进的Predenergy组合模型"""
    
    def __init__(self, config):
        super(PredenergyModel, self).__init__()
        self.config = config
        
        # STDM components
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
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
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        
        # 改进的连接层
        self.connection = PredenergyAdaptiveConnection(
            stdm_output_dim=config.d_model,
            motse_input_dim=config.motse_config.hidden_size,
            config=config
        )
        
        # 完整的MoTSE模型
        self.motse_model = MoTSEModel(config.motse_config)
        
        # MoTSE输出层
        self.motse_output_layer = MoTSEOutputLayer(
            hidden_size=config.motse_config.hidden_size,
            horizon_length=config.horizon,
            input_size=config.motse_config.input_size
        )
        
        # 最终输出投影
        self.final_output = nn.Linear(config.motse_config.hidden_size, config.pred_len)
        
        # 可选的额外处理层
        if config.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.pred_len)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input, return_attention=False):
        """
        Forward pass of the combined model
        
        Args:
            input: [batch_size, seq_len, n_vars] or dict with 'input_ids' key
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, pred_len, n_vars]
            L_importance: Importance weights from STDM
            attention_weights: (optional) Attention weights
        """
        # Handle input format - support both direct input and dict format
        if isinstance(input, dict):
            if 'input_ids' in input:
                input_data = input['input_ids']
            else:
                raise ValueError("Input dict must contain 'input_ids' key")
        else:
            input_data = input
            
        # STDM forward pass
        if self.CI:
            channel_independent_input = rearrange(input_data, 'b l n -> (b n) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input_data.shape[0])
        else:
            temporal_feature, L_importance = self.cluster(input_data)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)
            stdm_output = channel_group_feature
        else:
            stdm_output = temporal_feature
            attention = None
        
        # 连接层处理
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
        
        # 获取MoTSE的隐藏状态和router_logits
        motse_hidden_states = motse_outputs.last_hidden_state
        router_logits = motse_outputs.router_logits if hasattr(motse_outputs, 'router_logits') else None
        
        # MoTSE输出层
        motse_predictions = self.motse_output_layer(motse_hidden_states)
        
        # 最终输出投影
        final_output = self.final_output(motse_hidden_states)
        
        # 可选的层归一化
        if hasattr(self, 'final_norm'):
            final_output = self.final_norm(final_output)
        
        # 重塑输出以匹配预期格式 [batch_size, pred_len, n_vars]
        final_output = rearrange(final_output, 'b n d -> b d n')
        
        # 应用RevIN反归一化（如果需要）
        if hasattr(self.cluster, 'revin'):
            final_output = self.cluster.revin(final_output, "denorm")
        
        # 准备返回结果
        outputs = {
            'predictions': final_output,
            'L_importance': L_importance,
            'motse_predictions': motse_predictions,
            'motse_hidden_states': motse_hidden_states,  # 添加MoTSE隐藏状态
            'router_logits': router_logits  # 添加router_logits
        }
        
        if return_attention:
            outputs['attention_weights'] = attention
            outputs['motse_attention_weights'] = motse_outputs.attentions
        
        return outputs


class PredenergyForPrediction(nn.Module, TSGenerationMixin):
    """用于预测的完整Predenergy模型，支持生成和自回归预测"""
    
    def __init__(self, config):
        super(PredenergyForPrediction, self).__init__()
        self.config = config
        self.model = PredenergyModel(config)
        
        # 损失函数
        self.loss_function = torch.nn.HuberLoss(reduction='none', delta=2.0)
        
        # 多尺度预测头 - 支持多个预测长度
        self.multi_scale_heads = nn.ModuleList([
            nn.Linear(config.motse_config.hidden_size, horizon_length * config.motse_config.input_size)
            for horizon_length in config.motse_config.horizon_lengths
        ])
        
        # 预测长度映射
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.motse_config.horizon_lengths):
            self.horizon_length_map[horizon_length] = i
        
        # 生成配置
        self.generation_config = getattr(config, 'generation_config', None)
        if self.generation_config is None:
            from transformers import GenerationConfig
            self.generation_config = GenerationConfig(
                max_length=config.pred_len,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.0,
                pad_token_id=0,
                eos_token_id=1,
            )
        
        # 设备设置
        self.device = torch.device('cpu')
    
    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        self.device = device
        return self
    
    def forward(
        self, 
        input, 
        labels=None, 
        loss_masks=None,
        max_horizon_length=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    ):
        """
        完整的前向传播，支持训练和推理
        
        Args:
            input: [batch_size, seq_len, n_vars] or dict with 'input_ids' key
            labels: [batch_size, pred_len, n_vars] (optional)
            loss_masks: [batch_size, pred_len, n_vars] (optional)
            max_horizon_length: 最大预测长度 (optional)
            use_cache: 是否使用缓存 (optional)
            output_attentions: 是否输出注意力权重 (optional)
            output_hidden_states: 是否输出隐藏状态 (optional)
            return_dict: 是否返回字典格式 (optional)
        
        Returns:
            outputs: 包含预测结果和损失的字典
        """
        # Handle input format - support both direct input and dict format
        if isinstance(input, dict):
            if 'input_ids' in input:
                input_data = input['input_ids']
            elif 'inputs' in input:
                input_data = input['inputs']
            else:
                raise ValueError("Input dict must contain 'input_ids' or 'inputs' key")
        else:
            input_data = input
            
        # 模型前向传播
        model_outputs = self.model(input_data, return_attention=output_attentions)
        motse_hidden_states = model_outputs.get('motse_hidden_states', None)
        
        # 多尺度预测
        predictions = None
        motse_predictions = []
        
        if motse_hidden_states is not None:
            if labels is not None:
                # 训练模式：计算所有尺度的损失
                for lm_head, horizon_length in zip(self.multi_scale_heads, self.config.motse_config.horizon_lengths):
                    one_predictions = lm_head(motse_hidden_states)
                    motse_predictions.append(one_predictions)
                    if predictions is None:
                        predictions = one_predictions
            else:
                # 推理模式：选择合适的预测长度
                if max_horizon_length is None:
                    horizon_length = self.config.motse_config.horizon_lengths[0]
                    max_horizon_length = horizon_length
                else:
                    horizon_length = self.config.motse_config.horizon_lengths[0]
                    for h in self.config.motse_config.horizon_lengths[1:]:
                        if h > max_horizon_length:
                            break
                        else:
                            horizon_length = h
                
                lm_head = self.multi_scale_heads[self.horizon_length_map[horizon_length]]
                predictions = lm_head(motse_hidden_states)
                motse_predictions = [predictions]
                
                if horizon_length > max_horizon_length:
                    predictions = predictions[:, :, :self.config.motse_config.input_size * max_horizon_length]
        else:
            # 如果没有MoTSE隐藏状态，使用模型的预测结果
            predictions = model_outputs['predictions']
            motse_predictions = model_outputs.get('motse_predictions', None)
        
        # 重塑预测结果
        if predictions is not None:
            batch_size, seq_len, _ = predictions.shape
            predictions = predictions.view(batch_size, seq_len, -1, self.config.motse_config.input_size)
            predictions = predictions.transpose(-2, -1)  # [batch_size, seq_len, input_size, horizon_length]
        
        loss = None
        aux_loss = None
        
        if labels is not None:
            # Handle labels format - support both direct labels and dict format
            if isinstance(labels, dict):
                if 'labels' in labels:
                    labels_data = labels['labels']
                elif 'labels' in input:
                    labels_data = input['labels']
                else:
                    raise ValueError("Labels dict must contain 'labels' key")
            else:
                labels_data = labels
                
            # 计算自回归损失
            ar_loss = 0.0
            for i, (lm_head, horizon_length) in enumerate(zip(self.multi_scale_heads, self.config.motse_config.horizon_lengths)):
                if i < len(motse_predictions):
                    one_predictions = motse_predictions[i]
                    one_loss = self._calc_ar_loss(one_predictions, labels_data, loss_masks, horizon_length)
                    ar_loss += one_loss
            
            loss = ar_loss / len(self.config.motse_config.horizon_lengths)
            
            # 计算辅助损失（如果启用）
            if self.config.apply_aux_loss:
                if 'router_logits' in model_outputs:
                    router_logits = model_outputs['router_logits']
                    aux_loss = self._calc_load_balancing_loss(router_logits, loss_masks)
                    loss += self.config.router_aux_loss_factor * aux_loss
        
        # 准备返回结果
        outputs = {
            'loss': loss,
            'aux_loss': aux_loss,
            'predictions': predictions,
            'L_importance': model_outputs['L_importance'],
            'motse_predictions': motse_predictions
        }
        
        if output_attentions:
            outputs['attention_weights'] = model_outputs.get('attention_weights', None)
            outputs['motse_attention_weights'] = model_outputs.get('motse_attention_weights', None)
        
        if output_hidden_states:
            outputs['hidden_states'] = model_outputs.get('hidden_states', None)
        
        return outputs
    
    def _calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        """计算自回归损失"""
        if len(labels.shape) == 2:
            labels = labels.unsqueeze(dim=-1)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks = loss_masks.unsqueeze(dim=-1)
        
        # 确保设备一致
        labels = labels.to(predictions.device)
        if loss_masks is not None:
            loss_masks = loss_masks.to(predictions.device)
        
        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(batch_size, seq_len, horizon_length, -1)
            
            # 填充标签以匹配预测长度
            labels = F.pad(labels.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)
            
            if loss_masks is not None:
                loss_masks = F.pad(loss_masks.transpose(-1, -2), (0, horizon_length - 1), mode='constant', value=0)
                loss_masks = loss_masks.unfold(dimension=-1, size=horizon_length, step=1)
                loss_masks = loss_masks.permute(0, 2, 3, 1)
        else:
            shift_predictions = predictions
            shift_labels = labels
        
        # 计算损失
        losses = self.loss_function(shift_predictions, shift_labels)
        
        if loss_masks is not None:
            losses = losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = torch.mean(losses)
        
        return loss
    
    def _calc_load_balancing_loss(self, router_logits, attention_mask=None):
        """计算负载均衡损失"""
        if router_logits is None or not isinstance(router_logits, (tuple, list)) or router_logits[0] is None:
            return torch.tensor(0.0, device=self.device)
        
        compute_device = router_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in router_logits], dim=0)
        
        routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, self.config.motse_config.num_experts_per_tok, dim=-1)
        expert_mask = F.one_hot(selected_experts, self.config.motse_config.num_experts)
        
        if attention_mask is None:
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
            
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, 2, self.config.motse_config.num_experts))
                .reshape(-1, 2, self.config.motse_config.num_experts)
                .to(compute_device)
            )
            
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )
            
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, self.config.motse_config.num_experts))
                .reshape(-1, self.config.motse_config.num_experts)
                .to(compute_device)
            )
            
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )
        
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))
        return overall_loss * self.config.motse_config.num_experts
    
    def predict(self, input, max_horizon_length=None):
        """预测接口"""
        with torch.no_grad():
            # Handle input format
            if isinstance(input, dict):
                if 'input_ids' in input:
                    input_data = input
                else:
                    raise ValueError("Input dict must contain 'input_ids' key")
            else:
                input_data = {'input_ids': input}
                
            outputs = self.forward(
                input=input_data,
                max_horizon_length=max_horizon_length,
                return_dict=True
            )
            return outputs['predictions']
    
    def generate(self, input, max_length=None, **kwargs):
        """生成接口，支持自回归生成"""
        if max_length is None:
            max_length = self.config.pred_len
        
        # Handle input format
        if isinstance(input, dict):
            if 'input_ids' in input:
                input_data = input
            else:
                raise ValueError("Input dict must contain 'input_ids' key")
        else:
            input_data = {'input_ids': input}
        
        # 设置生成参数
        generation_kwargs = {
            'max_length': max_length,
            'do_sample': kwargs.get('do_sample', False),
            'temperature': kwargs.get('temperature', 1.0),
            'top_p': kwargs.get('top_p', 1.0),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.0),
            'pad_token_id': kwargs.get('pad_token_id', 0),
            'eos_token_id': kwargs.get('eos_token_id', 1),
        }
        
        # 使用TSGenerationMixin的生成方法
        return self._greedy_search(input_data, **generation_kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """为生成准备输入"""
        # Handle input format - input_ids can be either tensor or dict
        if isinstance(input_ids, dict):
            if 'input_ids' in input_ids:
                input_tensor = input_ids['input_ids']
            else:
                raise ValueError("Input dict must contain 'input_ids' key")
        else:
            input_tensor = input_ids
            
        # 这里需要根据实际的输入格式进行调整
        if past_key_values is not None:
            # 处理缓存的情况
            if isinstance(past_key_values, dict):
                cache_length = past_key_values.get('cache_length', 0)
            else:
                cache_length = past_key_values[0][0].shape[2] if past_key_values else 0
            
            if attention_mask is not None and attention_mask.shape[1] > input_tensor.shape[1]:
                input_tensor = input_tensor[:, -(attention_mask.shape[1] - cache_length):]
            elif cache_length < input_tensor.shape[1]:
                input_tensor = input_tensor[:, cache_length:]
        
        # 创建位置ID
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_tensor.shape[1]:]
        
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
    
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, horizon_length=1, **kwargs):
        """更新生成过程中的模型参数"""
        # 更新past_key_values
        if hasattr(outputs, 'past_key_values'):
            model_kwargs["past_key_values"] = outputs.past_key_values
        
        # 更新attention_mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], horizon_length))], dim=-1
            )
        
        # 更新position_ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, position_ids[:, -1].unsqueeze(-1) + horizon_length], dim=-1
            )
        
        return model_kwargs 