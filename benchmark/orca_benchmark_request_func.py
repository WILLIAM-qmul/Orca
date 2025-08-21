"""
Orca后端请求处理函数
"""
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import json


@dataclass
class RequestFuncInput:
    """请求输入数据结构"""
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None


@dataclass
class RequestFuncOutput:
    """请求输出数据结构"""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    tpot: float = 0.0  # Time per output token
    prompt_len: int = 0
    error: str = ""


async def async_request_orca(
    request_func_input: RequestFuncInput,
    pbar: Optional[any] = None,
) -> RequestFuncOutput:
    """向Orca后端发送异步请求"""
    api_url = request_func_input.api_url
    
    # Orca使用的是/generate端点
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "prompt": request_func_input.prompt,
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        st = time.perf_counter()
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    # Orca返回的是简单的JSON响应，不是流式
                    result = await response.json()
                    end_time = time.perf_counter()
                    
                    # 计算总延迟
                    output.latency = end_time - st
                    
                    # 提取生成的文本
                    if "response" in result:
                        output.generated_text = result["response"]
                        output.success = True
                        
                        # 计算生成的token数量（简单估算）
                        output.output_tokens = len(output.generated_text.split())
                        
                        # 对于非流式响应，TTFT就是总延迟
                        output.ttft = output.latency
                        
                        # TPOT计算（总时间除以token数）
                        if output.output_tokens > 0:
                            output.tpot = output.latency / output.output_tokens
                        
                    else:
                        output.error = "No response field in result"
                        output.success = False
                        
                else:
                    output.error = f"HTTP {response.status}: {response.reason}"
                    output.success = False
                    
        except Exception as e:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
            
        return output


# 后端请求函数映射
ASYNC_REQUEST_FUNCS = {
    "orca": async_request_orca,
}

# Orca兼容后端列表
ORCA_COMPATIBLE_BACKENDS = ["orca"]