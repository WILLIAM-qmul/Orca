from typing import Optional
from .llm import LLM
from .opt_decoder import OrcaOPTModel
from transformers.models.opt.configuration_opt import OPTConfig
from transformers import AutoTokenizer
import torch

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
        
    def generate(self, prompt: str) -> str:
        """Generate text for the given prompts using the modified OPT model with selective batching.

        Args:
            prompts str: Input prompt in string format

        Returns:
            str: Generated text in string format
        """
        
        response = self.model.generate(prompt)
        return response
    
    def batch_process(self, prompts: list[str], max_generation_length: int = 1) -> list[str]:
        """Process a batch of prompts using the modified OPT model with selective batching.

        Args:
            prompts (list[str]): List of prompts to process

        Returns:
            list[str]: List of generated texts
        """
        # first we need to tokenize the prompts
        encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device).bool()
        batch_size = input_ids.shape[0]
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        generated_tokens = [[] for _ in range(batch_size)]  # Store generated tokens per sequence
        
        # Initialize past_key_values
        past_key_values = None

        # Initialize sequence lengths
        sequence_lengths = attention_mask.sum(dim=1).tolist()

        # Generation loop
        for step in range(max_generation_length):
            # Prepare model inputs
            # At each step, we only need to input the last generated token (incremental decoding)
            if past_key_values is None:
                # First step, use the full input_ids and attention_mask
                model_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                    'use_cache': True,
                }
            else:
                # Subsequent steps, use only the last generated token
                next_tokens = input_ids[:, -1].unsqueeze(-1)  # Shape: [batch_size, 1]
                model_inputs = {
                    'input_ids': next_tokens,
                    'attention_mask': torch.ones_like(next_tokens, device=self.device),  # Only the new token
                    'past_key_values': past_key_values,
                    'use_cache': True,
                }

            # Forward pass
            outputs = self.model(**model_inputs)

            # Get logits for the last token
            logits = outputs.last_hidden_state  # Shape: [total_tokens, hidden_size]
            hidden_size = logits.size(-1)

            # Since we processed sequences individually, we need to map back the logits to sequences
            # For simplicity, we assume that outputs.last_hidden_state contains the logits for the new tokens
            # We'll reshape the logits to [batch_size, 1, hidden_size]
            # Note: In practice, you may need to adjust this based on how hidden_states are returned

            # In this simplified example, we'll simulate the output logits
            logits = logits[-batch_size:]  # Get the last batch_size logits
            logits = logits.view(batch_size, -1, hidden_size)  # Shape: [batch_size, seq_len=1, hidden_size]

            # Compute probabilities and sample the next token (e.g., using greedy decoding)
            next_token_logits = logits[:, -1, :]  # Shape: [batch_size, hidden_size]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)  # Shape: [batch_size]

            # Add generated tokens to input_ids
            input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1)  # Update input_ids

            # Update attention_mask
            new_attention_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

            # Update sequence_lengths
            sequence_lengths = [length + 1 if unfinished_sequences[i] else length for i, length in enumerate(sequence_lengths)]

            # Update past_key_values
            past_key_values = outputs.past_key_values

            # Check for eos_token_id and update unfinished_sequences
            for i, token_id in enumerate(next_token_ids):
                if token_id.item() == self.tokenizer.eos_token_id:
                    unfinished_sequences[i] = False

                if unfinished_sequences[i]:
                    generated_tokens[i].append(token_id.item())

            # If all sequences are finished, break the loop
            if not unfinished_sequences.any():
                break

            # Optional: Remove finished sequences from further computation
            # For sequences that are finished, we can skip updating them
            # This would involve more complex indexing and is omitted for simplicity

        # Decode generated tokens to text
        generated_texts = []
        for i in range(batch_size):
            # Get the tokens generated for this sequence
            generated_sequence = input_ids[i, :].tolist()
            # Decode the tokens (excluding the prompt)
            prompt_length = encoded_inputs['input_ids'][i].size(0)
            generated_tokens = generated_sequence[prompt_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

        
        
        
            
            
            
            