from typing import Optional
import torch    


class AttentionKVManager:
    def __init__(self):
        """Caches the attention keys and values in between iterations of token generation."""
        self.cache = {}
        
    def get(self, key: str) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve the attention keys and values for a given key.
        
        Args:
            key (str): The key to retrieve the attention keys and values.
        
        Returns:
            Optional[tuple[torch.Tensor, torch.Tensor]]: The attention keys and values or None if key is not found.
        """
        return self.cache.get(key, None)
    
    def store(self, request_id: str, past_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store the attention keys and values for a given key.
        
        Args:
            key (str): The request id to store the attention keys and values.
            attn_keys (torch.Tensor): The attention keys that we want to cache.
            attn_values (torch.Tensor): The attention values that we want to cache.
        """
        self.cache[request_id] = past_key_values
        
    def delete(self, key: str) -> None:
        """Delete the attention keys and values for a given request_id.
        
        Args:
            key (str): The key to delete from the cache.
            
        Raises:
            KeyError: If the key is not found in the cache.
        """
        if key in self.cache:
            del self.cache[key]
        else:
            raise KeyError(f"Key '{key}' not found in the cache.")
    