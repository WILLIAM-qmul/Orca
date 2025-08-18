from typing import Optional # 导入可选类型注解
import torch    # 导入torch库，用于张量操作


class AttentionKVManager:
    def __init__(self):
        """Caches the attention keys and values in between iterations of token generation."""
        self.cache = {} # 初始化缓存字典，用于存储每个请求的KV对
        
    def get(self, key: str) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve the attention keys and values for a given key.
        
        Args:
            key (str): The key to retrieve the attention keys and values.
        
        Returns:
            Optional[tuple[torch.Tensor, torch.Tensor]]: The attention keys and values or None if key is not found.
        """
        return self.cache.get(key, None)  # 从缓存中获取指定key的KV对，若不存在返回None
    
    def store(self, request_id: str, past_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store the attention keys and values for a given key.
        
        Args:
            key (str): The request id to store the attention keys and values.
            attn_keys (torch.Tensor): The attention keys that we want to cache.
            attn_values (torch.Tensor): The attention values that we want to cache.
        """
        self.cache[request_id] = past_key_values # 将KV对存入缓存，key为request_id
        
    def delete(self, key: str) -> None:
        """Delete the attention keys and values for a given request_id.
        
        Args:
            key (str): The key to delete from the cache.
            
        Raises:
            KeyError: If the key is not found in the cache.
        """
        if key in self.cache: # 如果key在缓存中
            del self.cache[key] # 删除该key对应的缓存
        else:
            raise KeyError(f"Key '{key}' not found in the cache.") # 否则抛出KeyError异常
    