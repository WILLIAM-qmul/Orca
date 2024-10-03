from enum import Enum
from pydantic import BaseModel

class RequestState(Enum):
    INITIATION = 1
    RUNNING = 2
    INCREMENT = 3
    COMPLETED = 4

### Pydantic models
class Request(BaseModel): 
    def __init__(self,  prompt: str, request_id: int = 0, max_tokens: int = 100):
        self.state: RequestState | None = RequestState.INITIATION
        self.max_tokens: int = max_tokens
        self.prompt: str = prompt
        self.request_id: int = request_id
        self.response: str = ""
        self.tokens_generated: int = 0
        
class Batch_Item(BaseModel):
    prompt: str
    request_id: int
    batch_id: int
    
        
class Batch(BaseModel):
    requests: list[Batch_Item]

class Prompt_Request(BaseModel):
    prompt: str
    
class Batch_Prompt_Request(BaseModel):
    prompts: list[str]