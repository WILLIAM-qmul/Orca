import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding, OPTPreTrainedModel, BaseModelOutputWithPast

# Assume that these classes and functions are defined elsewhere in your codebase
# from transformers import OPTConfig, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast

# Placeholder for the attention classes
OPT_ATTENTION_CLASSES = {
    'default': nn.MultiheadAttention  # Use PyTorch's MultiheadAttention for simplicity
    # Add other attention implementations as needed
}

# Modified OPTDecoderLayer with selective batching
class OrcaOPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Assuming 'default' uses nn.MultiheadAttention
        self.self_attn = nn.MultiheadAttention(self.embed_dim, num_heads=12, dropout=config.dropout, batch_first=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

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
        Modified forward method to implement selective batching.

        Args:
            hidden_states (`torch.FloatTensor`): concatenated hidden states of all sequences, shape `(total_tokens, embed_dim)`
            sequence_lengths (`List[int]`): lengths of each sequence in the batch
            attention_masks (`List[Optional[torch.Tensor]]`): attention masks for each sequence
            layer_head_masks (`List[Optional[torch.Tensor]]`): head masks for each sequence
            past_key_values (`List[Optional[Tuple[torch.Tensor]]]`): past key values for each sequence
            position_ids_list (`List[Optional[torch.LongTensor]]`): position IDs for each sequence
            [Other arguments remain the same...]
        """

        residual = hidden_states

        # Non-Attention Operations (process all tokens together)
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Split hidden_states per sequence
        # Changed!!!!
        # _____________________________________________________
        # We split hidden_states according to sequence_lengths
        
        hidden_states_list = torch.split(hidden_states, sequence_lengths)
        # _____________________________________________________
        # Process Attention per sequence
        attention_outputs = []
        self_attn_weights_list = []
        present_key_value_list = []

        # Change: Loop over each sequence and process attention individually
        # TODO: Potentially refactor into a lambda function to parallelize
        for idx, hs in enumerate(hidden_states_list):
            seq_len = hs.size(0) # hs has shape [seq_len, embed_dim]
            # Generate causal mask (attn_mask) of shape [seq_len, seq_len]
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(hs.device)
            
            # Prepare key_padding_mask of shape [batch_size=1, seq_len]
            key_padding_mask = (attention_masks[idx] == 0)  # True where padding tokens are
            # Ensure key_padding_mask is of dtype torch.bool
            key_padding_mask = key_padding_mask.to(torch.bool)
            
            # prepare past_key_values
            past_kv = past_key_values[idx] if past_key_values is not None else None
            if past_kv is not None:
                past_key, past_value = past_kv
            else:
                past_key, past_value = None, None
            
            hs = hs.unsqueeze(0)  # Add batch dimension
            
            # perform self-attention (not batched)
            attn_output, attn_weights, present_key_value = self._self_attention_with_past(
                hs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, past_key=past_key, past_value=past_value, need_weights=output_attentions, use_cache=use_cache 
            )
            
            attn_output = attn_output.squeeze(0)  # Remove batch dimension
            attention_outputs.append(attn_output)

            if output_attentions:
                self_attn_weights_list.append(attn_weights)

            # For simplicity, we are not implementing caching here
            if use_cache:
                present_key_value_list.append(present_key_value)

        # Concatenate the outputs from each sequence back into a single tensor
        ## Changed: 
        # Concatenate attention outputs from all sequences
        # _____________________________________________________
        hidden_states = torch.cat(attention_outputs, dim=0)
        # _____________________________________________________
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Continue with the rest of the operations
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected Layers
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights_list,)
        if use_cache:
            outputs += (present_key_value_list,)

        return outputs
    
    def _self_attention_with_past(self, hidden_states, attn_mask: torch.Tensor, key_padding_mask, past_key=None, past_value=None, need_weights=False, use_cache=False):
        """Helper function to perform self-attention with past key and value tensors

        Args:
            hidden_states (_type_): hidden states of shape [batch_size, seq_len, embed_dim] which are the layers' input
            attn_mask (_type_): attention mask used for masking future tokens
            key_padding_mask (_type_): key padding mask used to mask padding tokens
            past_key (_type_, optional): Cached past key. Defaults to None.
            past_value (_type_, optional): Cached past value. Defaults to None.
            need_weigths (bool, optional): setting for needs weights. Defaults to False.
            use_cache (bool, optional): setting to check if we are using cache. Defaults to False.

        Returns:
            _type_: _description_
        """
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
            
            # update key_padding_mask
            if key_padding_mask is not None:
                past_key_padding_mask = torch.zeros((1, total_sequence_length), dtype=torch.bool, device=hidden_states.device)
                key_padding_mask = torch.cat([past_key_padding_mask], dim=1)
        else:
            target_seq_len = hidden_states.size(1)
            src_seq_len = hidden_states.size(1)
            attn_mask = torch.triu(torch.full((target_seq_len, src_seq_len), float('-inf')), diagonal=1).to(hidden_states.device)
            

        attn_output, attn_weights = self.self_attn(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
        
        return attn_output, attn_weights, present_key_value
            
            
        

# Modified OPTDecoder with selective batching
class OrcaOPTDecoder(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OrcaOPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()  # Assume this is handled elsewhere

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
        # [Most of the preprocessing code remains the same]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
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

        past_key_values_length = 0  # Simplified for this example

        # Prepare per-sequence inputs
        sequence_lengths = attention_mask.sum(dim=1).tolist()
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # Compute hidden_states per sequence
        hidden_states_list = [
            inputs_embeds[i, :seq_len] + pos_embeds[i, :seq_len]
            for i, seq_len in enumerate(sequence_lengths)
        ]

        # Concatenate all hidden_states
        # Changed: Concatenate hidden_states from all sequences
        # _____________________________________________________
        hidden_states = torch.cat(hidden_states_list, dim=0)  # Shape: [total_tokens, hidden_size]
        # _____________________________________________________

        # Prepare attention masks per sequence
        attention_masks_list = [
            attention_mask[i, :seq_len].unsqueeze(0)  # Shape: [1, seq_len]
            for i, seq_len in enumerate(sequence_lengths)
        ]

        layer_head_masks = [head_mask for _ in range(batch_size)] if head_mask is not None else [None] * batch_size

        position_ids_list = [
            position_ids[i, :seq_len] if position_ids is not None else None
            for i, seq_len in enumerate(sequence_lengths)
        ]

        # Initialize variables for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # decoder layers
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

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # [Postprocessing code remains the same]

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class OrcaOPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OrcaOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

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

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
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

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )