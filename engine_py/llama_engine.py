'''
detailed llama engine
'''
from typing import Optional
from .llama_decoder import OrcaLlamaModel
from .attention_kv_manager import AttentionKVManager
from models.request import Batch_Item
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools

class Llama_Engine():
    def __init__(self, model_path: str = "/home/lsl/wwg/Orca/Llama-2-7b-hf") -> None:
        self.model_path = model_path
        
        # Load configuration
        self.config = LlamaConfig.from_pretrained(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with custom architecture
        self.model = OrcaLlamaModel(self.config)
        
        # Load pretrained weights
        pretrained_model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")
        self.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
        
        # Add language modeling head
        self.lm_head = pretrained_model.lm_head
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move to device and convert to same dtype
        # 统一成 float32 先保证不再出现 half/float 混用
        self.model.to(self.device).to(torch.float32)
        self.lm_head.to(self.device).to(torch.float32)
        
        self.model.eval()
        self.attention_kv_manager = AttentionKVManager()
        
    def generate(self, prompt: str) -> str:
        """Generate text for the given prompt using the modified Llama model.

        Args:
            prompt str: Input prompt in string format

        Returns:
            str: Generated text in string format
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    # def batch_process(self, requests: list[Batch_Item], max_generation_length: int = 1) -> list[tuple[str, bool]]:
    #     """Process a batch of prompts using the modified Llama model with selective batching.

    #     Args:
    #         requests (list[Batch_Item]): List of requests to process
    #         max_generation_length (int): Maximum number of tokens to generate

    #     Returns:
    #         list[tuple[str, bool]]: List of tuples containing generated texts and completion flags
    #     """
    #     prompts = [request.prompt for request in requests]
    #     request_ids = [request.request_id for request in requests]
        
    #     # Tokenize prompts
    #     encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    #     input_ids = encoded_inputs['input_ids'].to(self.device)
    #     attention_mask = encoded_inputs['attention_mask'].to(self.device).bool()
    #     batch_size = input_ids.shape[0]
        
    #     # Get cached past_key_values
    #     batch_past_key_values = []
    #     for request_id in request_ids:
    #         past_key_values = self.attention_kv_manager.get(str(request_id))
    #         batch_past_key_values.append(past_key_values)
            
    #     per_request_input_ids = []
    #     per_request_attention_mask = []
    #     for i in range(batch_size):
    #         per_request_input_ids.append(input_ids[i])
    #         per_request_attention_mask.append(attention_mask[i])
        
    #     unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
    #     generated_tokens = [[] for _ in range(batch_size)]

    #     # Generation loop
    #     for step in range(max_generation_length):
    #         input_id_list = []
    #         attention_mask_list = []
    #         sequence_lengths = []
            
    #         for i in range(batch_size):
    #             past_kv = batch_past_key_values[i]
    #             if past_kv is None:
    #                 input_id_list.append(per_request_input_ids[i])
    #                 attention_mask_list.append(per_request_attention_mask[i])
    #             else:
    #                 # For incremental generation, only use the last token
    #                 last_token_id = per_request_input_ids[i][-1].unsqueeze(0)
    #                 input_id_list.append(last_token_id)
    #                 attention_mask_list.append(torch.ones_like(last_token_id, device=self.device))
    #             sequence_lengths.append(len(input_id_list[-1]))
            
    #         # Pad sequences
    #         input_ids_padded = pad_sequence(input_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #         attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            
    #         model_inputs = {
    #             'input_ids': input_ids_padded,
    #             'attention_mask': attention_mask_padded,
    #             'past_key_values': batch_past_key_values,
    #             'use_cache': True,
    #         }
            
    #         # Forward pass
    #         with torch.no_grad():
    #             outputs = self.model(**model_inputs)
                
    #             # Apply language modeling head
    #             # 保证 last_hidden_state 和 lm_head 权重 dtype 一致
    #             last_hidden_state = outputs.last_hidden_state
    #             if last_hidden_state.dtype != self.lm_head.weight.dtype:
    #                 last_hidden_state = last_hidden_state.to(self.lm_head.weight.dtype)
    #             logits = self.lm_head(last_hidden_state)
                
    #             # Get updated past_key_values
    #             outputs_past_key_values = outputs.past_key_values
    #             if outputs_past_key_values is not None:
    #                 num_layers = len(outputs_past_key_values)
    #                 batch_past_key_values = []
    #                 for i in range(batch_size):
    #                     seq_past_key_values = [outputs_past_key_values[layer_idx][i] for layer_idx in range(num_layers)]
    #                     batch_past_key_values.append(seq_past_key_values)
                
    #             # Get logits for the last token of each sequence
    #             last_token_indices = [sum(sequence_lengths[:i+1]) - 1 for i in range(len(sequence_lengths))]
    #             next_token_logits = logits.view(-1, logits.size(-1))[last_token_indices]
    #             next_token_ids = torch.argmax(next_token_logits, dim=-1)
                
    #             # Update sequences
    #             for i in range(batch_size):
    #                 per_request_input_ids[i] = torch.cat([per_request_input_ids[i], next_token_ids[i].unsqueeze(0)], dim=0)
    #                 per_request_attention_mask[i] = torch.cat([per_request_attention_mask[i], torch.ones(1, device=self.device, dtype=torch.bool)], dim=0)
                    
    #                 if unfinished_sequences[i]:
    #                     generated_tokens[i].append(next_token_ids[i].item())
                        
    #                     # Check if sequence is finished
    #                     if next_token_ids[i].item() == self.tokenizer.eos_token_id:
    #                         unfinished_sequences[i] = False
    #                         try:
    #                             self.attention_kv_manager.delete(str(request_ids[i]))
    #                         except KeyError:
    #                             pass  # Key not found, already deleted
    #                     else:
    #                         self.attention_kv_manager.store(str(request_ids[i]), batch_past_key_values[i])
                            
    #             # ----------- 新增：step 到达 68 时强制完成 -----------
    #             if step == 67:  # 因为step从0开始，step==67是第68步
    #                 for i in range(batch_size):
    #                     if unfinished_sequences[i]:
    #                         unfinished_sequences[i] = False
    #                         try:
    #                             self.attention_kv_manager.delete(str(request_ids[i]))
    #                         except KeyError:
    #                             pass
    #                 break

    #             # Break if all sequences are finished
    #             if not unfinished_sequences.any():
    #                 break

    #     # Decode generated tokens to text
    #     generated_texts = []
    #     for i in range(batch_size):
    #         text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
    #         request_completed = not unfinished_sequences[i]
    #         generated_texts.append((text, request_completed))
            
    #     return generated_texts
    def batch_process(self, requests: list[Batch_Item], max_generation_length: int = 1) -> list[tuple[str, bool]]:
        """Process a batch of prompts using the modified Llama model with selective batching.

        Args:
            requests (list[Batch_Item]): List of requests to process
            max_generation_length (int): Maximum number of tokens to generate

        Returns:
            list[tuple[str, bool]]: List of tuples containing generated texts and completion flags
        """
        prompts = [request.prompt for request in requests]
        request_ids = [request.request_id for request in requests]
        
        # Tokenize prompts
        encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device).bool()
        batch_size = input_ids.shape[0]
        
        # Get cached past_key_values
        batch_past_key_values = []
        for request_id in request_ids:
            past_key_values = self.attention_kv_manager.get(str(request_id))
            batch_past_key_values.append(past_key_values)
            
        per_request_input_ids = []
        per_request_attention_mask = []
        for i in range(batch_size):
            per_request_input_ids.append(input_ids[i])
            per_request_attention_mask.append(attention_mask[i])
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        generated_tokens = [[] for _ in range(batch_size)]

        # Generation loop
        for step in range(max_generation_length):
            input_id_list = []
            attention_mask_list = []
            sequence_lengths = []
            
            for i in range(batch_size):
                past_kv = batch_past_key_values[i]
                if past_kv is None:
                    input_id_list.append(per_request_input_ids[i])
                    attention_mask_list.append(per_request_attention_mask[i])
                else:
                    # For incremental generation, only use the last token
                    last_token_id = per_request_input_ids[i][-1].unsqueeze(0)
                    input_id_list.append(last_token_id)
                    attention_mask_list.append(torch.ones_like(last_token_id, device=self.device))
                sequence_lengths.append(len(input_id_list[-1]))
            
            # Pad sequences
            input_ids_padded = pad_sequence(input_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            
            model_inputs = {
                'input_ids': input_ids_padded,
                'attention_mask': attention_mask_padded,
                'past_key_values': batch_past_key_values,
                'use_cache': True,
            }
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                
                # Apply language modeling head
                last_hidden_state = outputs.last_hidden_state
                if last_hidden_state.dtype != self.lm_head.weight.dtype:
                    last_hidden_state = last_hidden_state.to(self.lm_head.weight.dtype)
                logits = self.lm_head(last_hidden_state)
                
                # Get updated past_key_values
                outputs_past_key_values = outputs.past_key_values
                if outputs_past_key_values is not None:
                    num_layers = len(outputs_past_key_values)
                    batch_past_key_values = []
                    for i in range(batch_size):
                        seq_past_key_values = [outputs_past_key_values[layer_idx][i] for layer_idx in range(num_layers)]
                        batch_past_key_values.append(seq_past_key_values)
                
                # Get logits for the last token of each sequence
                last_token_indices = [sum(sequence_lengths[:i+1]) - 1 for i in range(len(sequence_lengths))]
                next_token_logits = logits.view(-1, logits.size(-1))[last_token_indices]
                
                # 改进的采样策略：添加重复惩罚和温度采样
                next_token_ids = []
                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        # 应用重复惩罚
                        logits_i = next_token_logits[i].clone()
                        
                        # 对已生成的token应用重复惩罚
                        if len(generated_tokens[i]) > 0:
                            for token_id in set(generated_tokens[i][-10:]):  # 只对最近10个token应用惩罚
                                logits_i[token_id] /= 1.2  # 重复惩罚因子
                        
                        # 温度采样而非贪心解码
                        temperature = 0.8  # 降低温度以减少随机性
                        logits_i = logits_i / temperature
                        probs = torch.softmax(logits_i, dim=-1)
                        
                        # Top-k采样
                        k = 50  # 增加top-k数量以增加多样性
                        top_k_probs, top_k_indices = torch.topk(probs, k)
                        top_k_probs = top_k_probs / top_k_probs.sum()  # 重新归一化
                        
                        # 从top-k中采样
                        sampled_index = torch.multinomial(top_k_probs, 1)
                        next_token_id = top_k_indices[sampled_index]
                        next_token_ids.append(next_token_id)
                    else:
                        # 对于已完成的序列，使用pad token
                        next_token_ids.append(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
                
                next_token_ids = torch.cat(next_token_ids, dim=0)
                
                # Update sequences
                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        per_request_input_ids[i] = torch.cat([per_request_input_ids[i], next_token_ids[i].unsqueeze(0)], dim=0)
                        per_request_attention_mask[i] = torch.cat([per_request_attention_mask[i], torch.ones(1, device=self.device, dtype=torch.bool)], dim=0)
                        
                        generated_tokens[i].append(next_token_ids[i].item())
                        
                        # Check if sequence is finished (EOS token, max length, or reached 50 tokens)
                        if (next_token_ids[i].item() == self.tokenizer.eos_token_id or 
                            len(generated_tokens[i]) >= 50):  # 修改为50个token限制
                            unfinished_sequences[i] = False
                            try:
                                self.attention_kv_manager.delete(str(request_ids[i]))
                            except KeyError:
                                pass
                        else:
                            self.attention_kv_manager.store(str(request_ids[i]), batch_past_key_values[i])

                # Break if all sequences are finished
                if not unfinished_sequences.any():
                    break

        # Decode generated tokens to text
        generated_texts = []
        for i in range(batch_size):
            text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
            request_completed = not unfinished_sequences[i]
            generated_texts.append((text, request_completed))
            
        return generated_texts