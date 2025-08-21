from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import torch
from .attention_kv_manager import AttentionKVManager
from models.request import Batch_Item
import itertools
from torch.nn.utils.rnn import pad_sequence

class Llama_Engine:
    def __init__(self, model_path="/home/lsl/wwg/models/Llama-2-7b-hf"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16  # 使用半精度以节省显存
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            
        print(f"使用设备: {self.device}，精度: {self.dtype}")
        
        print("加载 Llama tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("加载 Llama 模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=self.dtype,
            device_map=self.device,  # 自动管理设备映射
            low_cpu_mem_usage=True   # 减少CPU内存使用
        )
        
        self.model.eval()  # 设置为评估模式
        self.attention_kv_manager = AttentionKVManager()
        print("Llama 引擎初始化完成")

    def batch_process(self, requests: list[Batch_Item], max_generation_length: int = 1) -> list[tuple[str, bool]]:
        prompts = [request.prompt for request in requests]
        request_ids = [request.request_id for request in requests]
        batch_size = len(prompts)
        
        # 编码输入
        encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        generated_tokens = [[] for _ in range(batch_size)]
        
        # 初始化每个序列的 past_key_values 为 None
        per_request_past_key_values = [None] * batch_size
        
        # 生成循环
        for step in range(max_generation_length):
            # 如果是第一步，使用完整输入；否则只使用上一步生成的token
            if step == 0:
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
            else:
                # 构造新的input_ids，只包含上一步生成的token
                new_input_ids = torch.cat([next_token_ids.unsqueeze(-1)], dim=0)
                # 更新attention_mask
                new_attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device)
                ], dim=1)
                
                model_inputs = {
                    "input_ids": new_input_ids,
                    "attention_mask": new_attention_mask,
                    "use_cache": True,
                }
            
            # 模型前向传递
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                
            # 获取预测下一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)
            
            # 更新生成的tokens
            for i in range(batch_size):
                if unfinished_sequences[i]:
                    generated_tokens[i].append(next_token_ids[i].item())
                    # 检查序列是否完成
                    if next_token_ids[i].item() == self.tokenizer.eos_token_id:
                        unfinished_sequences[i] = False
            
            # 如果所有序列都完成，跳出循环
            if not unfinished_sequences.any():
                break
        
        # 解码生成的token为文本
        generated_texts = []
        for i in range(batch_size):
            text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
            request_completed = not unfinished_sequences[i]
            generated_texts.append((text, request_completed))
            
        return generated_texts
    
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
