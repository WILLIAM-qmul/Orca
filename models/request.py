from enum import Enum
from pydantic import BaseModel, Field

class RequestState(Enum):
    INITIATION = 1
    RUNNING = 2
    INCREMENT = 3
    COMPLETED = 4

### Pydantic models
class Request(BaseModel): 
    state: RequestState | None = RequestState.INITIATION
    max_tokens: int = Field(default=100)
    prompt: str
    request_id: int = Field(default=0)
    response: str = Field(default="")
    tokens_generated: int = Field(default=0)
    
class Batch_Item(BaseModel):
    prompt: str
    request_id: int
        
class Batch(BaseModel):
    requests: list[Batch_Item]
    
class Batch_Response_Item(BaseModel):
    request_id: int
    generated_tokens: str
    request_completed: bool
    
class Batch_Response(BaseModel):
    responses: list[Batch_Response_Item]


class Prompt_Request(BaseModel):
    prompt: str
    