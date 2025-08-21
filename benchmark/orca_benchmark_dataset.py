"""
数据集采样框架，支持ShareGPT数据集采样
"""
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers import PreTrainedTokenizerBase


@dataclass
class SampleRequest:
    """
    表示单个推理请求的数据结构
    """
    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    chat_history: str = ""


class BenchmarkDataset(ABC):
    """基准测试数据集基类"""
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.dataset_path = dataset_path
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    @abstractmethod
    def load_data(self) -> None:
        """加载数据集数据"""
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(
        self, tokenizer: PreTrainedTokenizerBase, num_requests: int
    ) -> list[SampleRequest]:
        """从数据集中采样生成请求"""
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self, requests: list[SampleRequest], num_requests: int
    ) -> None:
        """如果请求数量不足，进行过采样"""
        if len(requests) < num_requests:
            # 重复采样直到达到所需数量
            while len(requests) < num_requests:
                requests.extend(requests[:min(len(requests), num_requests - len(requests))])


class ShareGPTDataset(BenchmarkDataset):
    """ShareGPT数据集实现"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        """加载ShareGPT数据集"""
        if self.dataset_path is None:
            raise ValueError("ShareGPT dataset path must be provided")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)

        # 过滤掉对话轮次少于2的条目
        self.data = [
            entry
            for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]

        # 设置随机种子并打乱数据
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list[SampleRequest]:
        """从ShareGPT数据集采样"""
        samples = []
        
        for entry in self.data:
            if len(samples) >= num_requests:
                break

            conversations = entry["conversations"]
            
            # 构建提示（通常使用第一轮用户输入）
            prompt = ""
            expected_output = ""
            
            # 查找用户输入和助手回复
            for i, conv in enumerate(conversations):
                if conv.get("from") == "human":
                    prompt = conv["value"]
                    # 查找对应的助手回复
                    if i + 1 < len(conversations) and conversations[i + 1].get("from") == "gpt":
                        expected_output = conversations[i + 1]["value"]
                        break

            if not prompt:
                continue

            # 计算prompt长度
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)

            # 计算期望输出长度
            if output_len is not None:
                expected_output_len = output_len
            else:
                if expected_output:
                    output_tokens = tokenizer.encode(expected_output, add_special_tokens=False)
                    expected_output_len = len(output_tokens)
                else:
                    expected_output_len = 128  # 默认长度

            samples.append(SampleRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=expected_output_len,
                chat_history=""
            ))

        # 如果样本数量不足，进行过采样
        self.maybe_oversample_requests(samples, num_requests)
        
        return samples[:num_requests]


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
) -> bool:
    """验证序列是否符合长度要求"""
    prompt_too_short = prompt_len < min_len
    output_too_short = output_len < min_len
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )