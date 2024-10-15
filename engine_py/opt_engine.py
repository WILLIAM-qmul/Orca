from typing import Optional
from .llm import LLM
from .opt_decoder import OrcaOPTModel
from .attention_kv_manager import AttentionKVManager
from models.request import Batch_Item
from transformers.models.opt.configuration_opt import OPTConfig
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools

class OPT_Engine():
    def __init__(self, model: Optional[OrcaOPTModel] = OrcaOPTModel(OPTConfig())) -> None:
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.attention_kv_manager = AttentionKVManager()
        
        
    def generate(self, prompt: str) -> str:
        """Generate text for the given prompts using the modified OPT model with selective batching.

        Args:
            prompts str: Input prompt in string format

        Returns:
            str: Generated text in string format
        """
        
        response = self.model.generate(prompt)
        return response
    
    def batch_process(self, requests: list[Batch_Item], max_generation_length: int = 1) -> list[tuple[str, bool]]:
        """Process a batch of prompts using the modified OPT model with selective batching.

        Args:
            prompts (list[str]): List of prompts to process

        Returns:
            list[tuple[str, bool]]: List of tuples containing generated texts (str) and a boolean flag indicating if the sequence is completed
        """
        # first we need to tokenize the prompts
        prompts = [request.prompt for request in requests]
        request_ids = [request.request_id for request in requests]
        encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device).bool()
        batch_size = input_ids.shape[0]
        
       
        
        # get cached past_key_values and store as a batch
        batch_past_key_values = []
        for request_id in request_ids:
            past_key_values = self.attention_kv_manager.get(request_id)
            batch_past_key_values.append(past_key_values)
            
        per_request_input_ids = []
        per_request_attention_mask = []
        for i in range(batch_size):
            per_request_input_ids.append(input_ids[i])
            per_request_attention_mask.append(attention_mask[i])
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        generated_tokens = [[] for _ in range(batch_size)]  # Store generated tokens per sequence

        # Generation loop
        for step in range(max_generation_length):
            # Prepare model inputs
            input_id_list = []
            attention_mask_list = []
            sequence_lengths = []
            
            for i in range(batch_size):
                past_kv = batch_past_key_values[i]
                if past_kv is None:
                    input_id_list.append(per_request_input_ids[i])
                    attention_mask_list.append(per_request_attention_mask[i])
                else:
                    last_token_id = per_request_input_ids[i][-1].unsqueeze(0)
                    input_id_list.append(last_token_id)
                    attention_mask_list.append(torch.ones_like(last_token_id, device=self.device))
                sequence_lengths.append(len(input_id_list[-1]))
            # pad sequences up to max length of the current batch
            input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask_padded,
                'past_key_values': batch_past_key_values,
                'use_cache': True,
            }
            
            # forward pass
            outputs = self.model(**model_inputs)
            
            # get the updated past_key_values
            outputs_past_key_values = outputs.past_key_values # list of layers with each layer having tuple of (key tensor, value tensor)
            num_layers = len(outputs_past_key_values)
            batch_past_key_values = []
            for i in range(batch_size):
                seq_past_key_values = [outputs_past_key_values[layer_idx][i] for layer_idx in range(num_layers)]
                batch_past_key_values.append(seq_past_key_values)
            
            
            
            # Get logits for the last token
            logits = outputs.last_hidden_state  # Shape: [total_tokens, hidden_size]
            last_token_index = [cum - 1 for cum in list(itertools.accumulate(sequence_lengths))]
            next_token_logits = logits[last_token_index, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)  # Shape: [batch_size]
            
            # update input_ids and attention_mask and cache the past_key_values
            for i in range(batch_size):
                per_request_input_ids[i] = torch.cat([per_request_input_ids[i], next_token_ids[i].unsqueeze(0)], dim=0)
                per_request_attention_mask[i] = torch.cat([per_request_attention_mask[i], torch.ones(1, device=self.device, dtype=torch.bool)], dim=0)
                
                if unfinished_sequences[i]:
                    generated_tokens[i].append(next_token_ids[i].item())
                    # Check if the sequence is finished
                    if next_token_ids[i].item() == self.tokenizer.eos_token_id:
                        unfinished_sequences[i] = False
                        self.attention_kv_manager.delete(request_ids[i])
                    else:
                        self.attention_kv_manager.store(request_ids[i], batch_past_key_values[i])
                        

            # Update input_ids with the new token
            # If all sequences are finished, break the loop
            if not unfinished_sequences.any():
                break


        # Decode generated tokens to text
        generated_texts = []
        for i in range(batch_size):
            # Decode the generated tokens into readable text
            text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
            request_completed = not unfinished_sequences[i]
            generated_texts.append((text, request_completed))
        return generated_texts

        
        
        
            
            
            
            