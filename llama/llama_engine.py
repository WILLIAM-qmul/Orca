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