import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

# Llama特有的RMSNorm实现
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm是Llama模型使用的归一化方式
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

# 修改的LlamaDecoderLayer实现选择性批处理
class OrcaLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 注意力机制
        self.self_attn = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_dropout if hasattr(config, "attention_dropout") else 0.0,
            batch_first=True
        )
        
        # Llama使用RMSNorm
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        # Llama使用SwiGLU激活函数的前馈网络
        self.mlp_dim = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.mlp_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.mlp_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_dim, self.hidden_size, bias=False)
        
        # 修复：使用 config.dropout 代替 config.attention_dropout 和 config.hidden_dropout
        self.attention_dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.hidden_dropout = config.dropout if hasattr(config, "dropout") else 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: List[int],
        attention_masks: List[Optional[torch.Tensor]],
        layer_head_masks: List[Optional[torch.Tensor]],
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_ids_list: List[Optional[torch.LongTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Llama解码器层的前向传播实现，支持选择性批处理
        """
        residual = hidden_states
        
        # 应用RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        
        # 分割隐藏状态
        hidden_states_list = torch.split(hidden_states, sequence_lengths)
        
        # 处理每个序列的注意力
        attention_outputs = []
        self_attn_weights_list = []
        present_key_value_list = []

        # 循环处理每个序列
        for idx, hs in enumerate(hidden_states_list):
            seq_len = hs.size(0)
            
            # 生成注意力掩码
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(hs.device)
            
            # 准备key_padding_mask
            key_padding_mask = None
            if attention_masks[idx] is not None:
                key_padding_mask = (attention_masks[idx] == 0)
                key_padding_mask = key_padding_mask.to(torch.bool)
            
            # 处理past_key_values
            past_kv = past_key_values[idx] if past_key_values is not None else None
            if past_kv is not None:
                past_key, past_value = past_kv
            else:
                past_key, past_value = None, None
            
            hs = hs.unsqueeze(0)  # 添加批次维度
            
            # 执行自注意力
            attn_output, attn_weights, present_key_value = self._self_attention_with_past(
                hs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, 
                past_key=past_key, past_value=past_value, 
                need_weights=output_attentions, use_cache=use_cache 
            )
            
            attn_output = attn_output.squeeze(0)  # 移除批次维度
            attention_outputs.append(attn_output)

            if output_attentions:
                self_attn_weights_list.append(attn_weights)

            if use_cache:
                present_key_value_list.append(present_key_value)

        # 连接所有序列的输出
        hidden_states = torch.cat(attention_outputs, dim=0)
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 前馈网络部分
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # SwiGLU激活
        gate_output = torch.sigmoid(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        hidden_states = gate_output * up_output
        hidden_states = self.down_proj(hidden_states)
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights_list,)
        if use_cache:
            outputs += (present_key_value_list,)

        return outputs

    def _self_attention_with_past(self, hidden_states, attn_mask, key_padding_mask, past_key=None, past_value=None, need_weights=False, use_cache=False):
        """处理自注意力逻辑，支持past_key_values缓存"""
        query = hidden_states
        key = hidden_states
        value = hidden_states
        
        if past_key is not None and past_value is not None:
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
            
            total_sequence_length = past_key.size(1) + hidden_states.size(1)
            
            target_seq_len = hidden_states.size(1)
            src_len = total_sequence_length
            attn_mask = torch.triu(torch.full((target_seq_len, src_len), float('-inf')), diagonal=1).to(attn_mask.device)
            
            # 更新key_padding_mask
            if key_padding_mask is not None:
                past_key_padding_mask = torch.zeros((1, total_sequence_length), dtype=torch.bool, device=hidden_states.device)
                key_padding_mask = torch.cat([past_key_padding_mask], dim=1)
        else:
            target_seq_len = hidden_states.size(1)
            src_seq_len = hidden_states.size(1)
            attn_mask = torch.triu(torch.full((target_seq_len, src_seq_len), float('-inf')), diagonal=1).to(hidden_states.device)

        attn_output, attn_weights = self.self_attn(
            query, key, value, attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
        
        return attn_output, attn_weights, present_key_value

# 修改的LlamaDecoder实现选择性批处理
class OrcaLlamaDecoder(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([OrcaLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 初始化权重
        self.post_init()
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # 创建因果注意力掩码
        # 这个方法在选择性批处理中会在每层单独处理
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检索input_ids和inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = input_shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        
        # 准备每个序列的输入
        sequence_lengths = attention_mask.sum(dim=1).tolist()
        
        # 计算position_ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, input_shape[-1] + past_key_values_length,
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        # 计算每个序列的hidden_states
        hidden_states_list = [
            inputs_embeds[i, :seq_len]
            for i, seq_len in enumerate(sequence_lengths)
        ]

        # 连接所有hidden_states
        hidden_states = torch.cat(hidden_states_list, dim=0)  # Shape: [total_tokens, hidden_size]

        # 准备每个序列的attention_masks
        attention_masks_list = [
            attention_mask[i, :seq_len].unsqueeze(0)  # Shape: [1, seq_len]
            for i, seq_len in enumerate(sequence_lengths)
        ]

        layer_head_masks = [head_mask for _ in range(batch_size)] if head_mask is not None else [None] * batch_size

        position_ids_list = [
            position_ids[i, :seq_len] if position_ids is not None else None
            for i, seq_len in enumerate(sequence_lengths)
        ]

        # 初始化输出变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 解码器层
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is None:
                past_key_values_layer = [None] * batch_size
            else:
                past_key_values_layer = []
                for pkv in past_key_values:
                    if pkv is None:
                        past_key_values_layer.append(None)
                    else:
                        past_key_values_layer.append(pkv[idx])

            layer_outputs = decoder_layer(
                hidden_states,
                sequence_lengths=sequence_lengths,
                attention_masks=attention_masks_list,
                layer_head_masks=layer_head_masks,
                past_key_values=past_key_values_layer,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_ids_list=position_ids_list,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                present_key_value = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache += (present_key_value,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class OrcaLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.model = OrcaLlamaDecoder(config)
        
        # 初始化权重
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用解码器
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs