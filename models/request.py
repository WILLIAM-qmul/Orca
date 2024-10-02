from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel

class RequestState(Enum):
    INITIATION = 1
    RUNNING = 2
    INCREMENT = 3
    COMPLETED = 4

# Request Object
@dataclass
class Request(): 
    def __init__(self,  prompt: str, request_id: int = 0, max_tokens: int = 100):
        self.state = RequestState.INITIATION
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.request_id = request_id
        self.response = ""
        self.tokens_generated = 0
        
        
### Pydantic models
class Scheduled_Iteration(BaseModel):
    requests: list[Request]

class Iteration_Responses(BaseModel):
    responses: list[str]

class Prompt_Request(BaseModel):
    prompt: str
    
class Batch_Prompt_Request(BaseModel):
    prompts: list[str]