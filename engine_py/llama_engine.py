'''
simple llama engine
'''
# from typing import Optional
# from .attention_kv_manager import AttentionKVManager
# from models.request import Batch_Item
# from transformers import AutoTokenizer, LlamaForCausalLM # 导入transformers的tokenizer和Llama模型
# import torch
# from torch.nn.utils.rnn import pad_sequence # 导入pad_sequence用于序列填充（本文件未直接用到）
# import gc # 导入gc用于垃圾回收

# class Llama_Engine():
#     def __init__(self, model_path: str = "/home/lsl/wwg/models/Llama-2-7b-hf") -> None:
#         self.model_path = model_path
        
#         # Setup device first
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             print(f"Using CUDA device")
#         else:
#             self.device = torch.device("cpu")
#             print(f"Using CPU device")
        
#         # Load tokenizer
#         print("Loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token # 如果没有pad_token，设置为eos_token
        
#         # Load model with memory optimizations
#         print("Loading model...")
#         try:
#             if torch.cuda.is_available():
#                 # 全部加载到GPU 0，不限制显存
#                 self.model = LlamaForCausalLM.from_pretrained(
#                     model_path,
#                     torch_dtype=torch.float16,
#                     device_map={"": 0},  # 强制全部放到GPU 0
#                     low_cpu_mem_usage=True,
#                 )
#             else:
#                 # For CPU, use smaller precision to reduce memory
#                 self.model = LlamaForCausalLM.from_pretrained(
#                     model_path,
#                     torch_dtype=torch.float32,
#                     low_cpu_mem_usage=True,
#                     device_map="cpu",
#                 )
#             # if torch.cuda.is_available():
#             #     # For CUDA, use fp16 and device_map for memory efficiency
#             #     self.model = LlamaForCausalLM.from_pretrained(
#             #         model_path,
#             #         torch_dtype=torch.float16,
#             #         device_map="auto",
#             #         low_cpu_mem_usage=True,
#             #         max_memory={0: "6GB"},  # Adjust based on your GPU memory
#             #     )
#             # else:
#             #     # For CPU, use smaller precision to reduce memory
#             #     self.model = LlamaForCausalLM.from_pretrained(
#             #         model_path,
#             #         torch_dtype=torch.float32,
#             #         low_cpu_mem_usage=True,
#             #         device_map="cpu",
#             #     )
#         except Exception as e:
#             print(f"Failed to load full model: {e}")
#             print("Trying to load with smaller configuration...")
#             # Fallback to smaller model or configuration
#             try:
#                 self.model = LlamaForCausalLM.from_pretrained(
#                     "microsoft/DialoGPT-medium",  # Fallback to smaller model
#                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                     device_map="auto" if torch.cuda.is_available() else "cpu",
#                 )
#                 print("Loaded fallback model: DialoGPT-medium")
#             except Exception as e2:
#                 print(f"Fallback also failed: {e2}")
#                 raise e2
        
#         self.model.eval() # 设置为推理模式
#         self.attention_kv_manager = AttentionKVManager() # 初始化KV缓存管理器
        
#         # Clear memory
#         gc.collect() # 垃圾回收
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache() # 清空GPU缓存
        
#         print("Model loaded successfully")
        
#     def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
#         """Generate text for the given prompt."""
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512) # 编码输入
#             if torch.cuda.is_available() and hasattr(self.model, 'device'):
#                 inputs = {k: v.to(self.model.device) for k, v in inputs.items()} # 移动到模型设备
#             else:
#                 inputs = {k: v.to(self.device) for k, v in inputs.items()} # 移动到CPU
            
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens, # 生成最大新token数
#                     do_sample=True, # 采样生成
#                     temperature=0.7, # 采样生成
#                     pad_token_id=self.tokenizer.eos_token_id, # 填充token
#                     use_cache=True, # 启用KV缓存
#                 )
            
#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # 解码输出
#             return response[len(prompt):].strip() # 返回去掉prompt部分的新生成内容
#         except Exception as e:
#             print(f"Error in generation: {e}")
#             return "Error generating response"
    
#     def batch_process(self, requests: list[Batch_Item], max_generation_length: int = 1) -> list[tuple[str, bool]]:
#         """Process a batch of prompts with memory optimization."""
#         try:
#             prompts = [request.prompt for request in requests] # 提取所有prompt
#             request_ids = [request.request_id for request in requests] # 提取所有请求ID
            
#             # Limit batch size to prevent OOM
#             if len(requests) > 2:
#                 requests = requests[:2]
#                 prompts = prompts[:2]
#                 request_ids = request_ids[:2]
            
#             # Tokenize with length limits
#             encoded_inputs = self.tokenizer(
#                 prompts, 
#                 return_tensors="pt", 
#                 padding=True, 
#                 truncation=True,
#                 max_length=256  # Limit input length
#             )
            
#             # Move to device
#             device = self.model.device if hasattr(self.model, 'device') else self.device
#             input_ids = encoded_inputs['input_ids'].to(device) # 输入ID移到设备
#             attention_mask = encoded_inputs['attention_mask'].to(device).bool() # 注意力mask移到设备
#             batch_size = input_ids.shape[0] # 批量大小
            
#             # Simple generation without complex KV caching for now
#             generated_texts = [] # 存储生成结果
            
#             for i in range(batch_size):
#                 try:
#                     single_input_ids = input_ids[i:i+1]
#                     single_attention_mask = attention_mask[i:i+1]
                    
#                     with torch.no_grad():
#                         outputs = self.model.generate(
#                             single_input_ids,
#                             attention_mask=single_attention_mask,
#                             max_new_tokens=max_generation_length,
#                             do_sample=True,
#                             temperature=0.7,
#                             pad_token_id=self.tokenizer.eos_token_id,
#                             use_cache=True,
#                         )
                    
#                     # Decode only the new tokens
#                     original_length = single_input_ids.shape[1]
#                     new_tokens = outputs[0, original_length:]
#                     text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
#                     # Simple completion check
#                     request_completed = (
#                         len(new_tokens) < max_generation_length or 
#                         self.tokenizer.eos_token_id in new_tokens
#                     )
                    
#                     generated_texts.append((text, request_completed))
                    
#                 except Exception as e:
#                     print(f"Error processing request {request_ids[i]}: {e}")
#                     generated_texts.append(("Error", True))
            
#             # Clear memory after processing
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
                
#             return generated_texts
            
#         except Exception as e:
#             print(f"Error in batch processing: {e}")
#             # Return error responses for all requests
#             return [("Error processing request", True) for _ in requests]

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
    def __init__(self, model_path: str = "/home/lsl/wwg/models/Llama-2-7b-hf") -> None:
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
        pretrained_model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
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
        
        # Move to device
        if not hasattr(self.model, 'device') or self.model.device != self.device:
            self.model.to(self.device)
        self.lm_head.to(self.device)
        
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
                logits = self.lm_head(outputs.last_hidden_state)
                
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
                next_token_ids = torch.argmax(next_token_logits, dim=-1)
                
                # Update sequences
                for i in range(batch_size):
                    per_request_input_ids[i] = torch.cat([per_request_input_ids[i], next_token_ids[i].unsqueeze(0)], dim=0)
                    per_request_attention_mask[i] = torch.cat([per_request_attention_mask[i], torch.ones(1, device=self.device, dtype=torch.bool)], dim=0)
                    
                    if unfinished_sequences[i]:
                        generated_tokens[i].append(next_token_ids[i].item())
                        
                        # Check if sequence is finished
                        if next_token_ids[i].item() == self.tokenizer.eos_token_id:
                            unfinished_sequences[i] = False
                            try:
                                self.attention_kv_manager.delete(str(request_ids[i]))
                            except KeyError:
                                pass  # Key not found, already deleted
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