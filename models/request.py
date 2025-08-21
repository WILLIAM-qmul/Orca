from enum import Enum # 导入枚举类型，用于定义请求
from threading import Event # 导入线程事件，用于线程间同步
from pydantic import BaseModel, Field, PrivateAttr

class RequestState(Enum): # 定义请求的状态枚举
    INITIATION = 1 # 初始状态
    RUNNING = 2 # 正在处理
    INCREMENT = 3 # 增量生成中（如多轮生成）
    COMPLETED = 4 # 已完成

### Pydantic models
class Request(BaseModel): # 单条请求的数据结构
    state: RequestState | None = RequestState.INITIATION # 当前请求状态，默认INITIATION
    max_tokens: int = Field(default=100) # 最大生成token数，默认100
    prompt: str # 用户输入的prompt
    request_id: int = Field(default=0) # 请求ID，默认0
    response: str = Field(default="") # 已生成的响应文本，默认空
    tokens_generated: int = Field(default=0) # 已生成的token数，默认0
    total_tokens_generated: int = Field(default=0) # 累计已生成的token总数，默认0
    def __init__(self, **data: any):
        super().__init__(**data)
        self._request_completed_signal: Event = Event()  # Event to signal completion # 创建线程事件，用于通知请求完成
        
    def wait_for_completion(self):
        self._request_completed_signal.wait() # 阻塞等待请求完成信号
    
    def mark_as_completed(self):
        self.state = RequestState.COMPLETED # 设置状态为已完成
        self._request_completed_signal.set() # 发送完成信号，唤醒等待线程
    

class Batch_Item(BaseModel): # 批量请求中的单条请求
    prompt: str # 用户输入的prompt
    request_id: int # 请求ID
        
class Batch(BaseModel):  # 批量请求的数据结构
    requests: list[Batch_Item] # 批量中的所有请求（Batch_Item列表）
    
class Batch_Response_Item(BaseModel): # 批量响应中的单条响应
    request_id: int # 请求ID
    generated_tokens: str # 生成的文本
    request_completed: bool # 本次生成后该请求是否已完成
    
class Batch_Response(BaseModel): # 批量响应的数据结构
    responses: list[Batch_Response_Item] # 批量中的所有响应（Batch_Response_Item列表）


class Prompt_Request(BaseModel): # 单条prompt请求的数据结构（用于API接口）
    prompt: str # 用户输入的prompt
    