"""
Orca在线服务吞吐量基准测试

使用方法:
python benchmark_serving.py \
    --backend orca \
    --model Llama-2-7b-hf \
    --dataset-name sharegpt \
    --dataset-path <path to dataset> \
    --request-rate <request_rate> \
    --num-prompts <num_prompts>
"""
import argparse
import asyncio
import gc
import random
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from orca_benchmark_dataset import ShareGPTDataset, SampleRequest
from orca_benchmark_request_func import (
    ASYNC_REQUEST_FUNCS,
    ORCA_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from orca_benchmark_utils import write_to_json, calculate_percentiles, get_timestamp_filename


@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    completed: int = 0
    total_input: int = 0
    total_output: int = 0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    percentiles_ttft_ms: List[tuple] = None
    mean_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    percentiles_tpot_ms: List[tuple] = None
    mean_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    percentiles_itl_ms: List[tuple] = None
    mean_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    percentiles_e2el_ms: List[tuple] = None
    mean_normalized_latency: float = 0.0


async def get_request(
    input_requests: List[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """按指定速率异步生成请求"""
    if request_rate == float("inf"):
        # 如果请求速率为无穷大，立即发送所有请求
        for request in input_requests:
            yield request
        return

    # 使用泊松过程或伽马分布生成请求间隔
    for i, request in enumerate(input_requests):
        if i == 0:
            # 第一个请求立即发送
            yield request
        else:
            # 计算下一个请求的间隔时间
            if burstiness == 1.0:
                # 泊松过程
                interval = random.expovariate(request_rate)
            else:
                # 伽马分布
                shape = 1.0 / burstiness
                scale = burstiness / request_rate
                interval = random.gammavariate(shape, scale)
            
            await asyncio.sleep(interval)
            yield request


def calculate_metrics(
    input_requests: List[SampleRequest],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: List[float],
) -> BenchmarkMetrics:
    """计算基准测试指标"""
    actual_output_lens = []
    total_input = 0
    completed = 0
    ttfts = []
    tpots = []
    itls = []
    e2els = []  # end-to-end latencies
    normalized_latencies = []

    for i in range(len(outputs)):
        if outputs[i].success:
            completed += 1
            total_input += outputs[i].prompt_len
            actual_output_lens.append(outputs[i].output_tokens)
            
            if outputs[i].ttft > 0:
                ttfts.append(outputs[i].ttft)
            if outputs[i].tpot > 0:
                tpots.append(outputs[i].tpot)
            if outputs[i].itl:
                itls.extend(outputs[i].itl)
            if outputs[i].latency > 0:
                e2els.append(outputs[i].latency)
                # 计算归一化延迟 (延迟/输出token数)
                if outputs[i].output_tokens > 0:
                    normalized_latencies.append(outputs[i].latency / outputs[i].output_tokens)

    if completed == 0:
        return BenchmarkMetrics()

    # 计算各项指标
    mean_normalized_latency = np.mean(normalized_latencies) if normalized_latencies else 0
    
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000 if ttfts else 0,
        std_ttft_ms=np.std(ttfts) * 1000 if ttfts else 0,
        median_ttft_ms=np.median(ttfts) * 1000 if ttfts else 0,
        percentiles_ttft_ms=[(p, np.percentile(ttfts, p) * 1000) for p in selected_percentiles] if ttfts else [],
        mean_tpot_ms=np.mean(tpots) * 1000 if tpots else 0,
        std_tpot_ms=np.std(tpots) * 1000 if tpots else 0,
        median_tpot_ms=np.median(tpots) * 1000 if tpots else 0,
        percentiles_tpot_ms=[(p, np.percentile(tpots, p) * 1000) for p in selected_percentiles] if tpots else [],
        mean_itl_ms=np.mean(itls) * 1000 if itls else 0,
        std_itl_ms=np.std(itls) * 1000 if itls else 0,
        median_itl_ms=np.median(itls) * 1000 if itls else 0,
        percentiles_itl_ms=[(p, np.percentile(itls, p) * 1000) for p in selected_percentiles] if itls else [],
        mean_e2el_ms=np.mean(e2els) * 1000 if e2els else 0,
        std_e2el_ms=np.std(e2els) * 1000 if e2els else 0,
        median_e2el_ms=np.median(e2els) * 1000 if e2els else 0,
        percentiles_e2el_ms=[(p, np.percentile(e2els, p) * 1000) for p in selected_percentiles] if e2els else [],
        mean_normalized_latency=mean_normalized_latency,
    )

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[SampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    selected_percentiles: List[float],
) -> dict:
    """执行基准测试"""
    if backend not in ASYNC_REQUEST_FUNCS:
        raise ValueError(f"Backend {backend} not supported")
    
    request_func = ASYNC_REQUEST_FUNCS[backend]

    # 测试连接
    print("Starting initial test run...")
    test_input = RequestFuncInput(
        model=model_id,
        prompt=input_requests[0].prompt,
        api_url=api_url,
        prompt_len=input_requests[0].prompt_len,
        output_len=input_requests[0].expected_output_len,
    )
    
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(f"Test request failed: {test_output.error}")
    else:
        print("Test request successful!")

    print(f"Request rate: {request_rate} req/s")
    print(f"Burstiness factor: {burstiness}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []

    async for request in get_request(input_requests, request_rate, burstiness):
        request_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
        )
        task = asyncio.create_task(request_func(request_input, pbar))
        tasks.append(task)

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # 计算指标
    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        selected_percentiles=selected_percentiles,
    )

    # 打印结果
    print("=" * 50)
    print(" Serving Benchmark Result ".center(50, "="))
    print("=" * 50)
    print(f"{'Successful requests:':<40} {metrics.completed:<10}")
    print(f"{'Benchmark duration (s):':<40} {benchmark_duration:<10.2f}")
    print(f"{'Total input tokens:':<40} {metrics.total_input:<10}")
    print(f"{'Total generated tokens:':<40} {metrics.total_output:<10}")
    print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
    print(f"{'Output token throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
    print(f"{'Total Token throughput (tok/s):':<40} {metrics.total_token_throughput:<10.2f}")
    print(f"{'Mean normalized latency (s/token):':<40} {metrics.mean_normalized_latency:<10.4f}")

    # 详细指标
    def print_metric_details(metric_name, mean_ms, std_ms, median_ms, percentiles):
        print(f"\n{metric_name}:")
        print(f"{'  Mean:':<38} {mean_ms:<10.2f} ms")
        print(f"{'  Std:':<38} {std_ms:<10.2f} ms") 
        print(f"{'  Median:':<38} {median_ms:<10.2f} ms")
        for p, value in percentiles:
            print(f"  P{p}:".ljust(38) + f" {value:<10.2f} ms")

    print_metric_details("Time to First Token", metrics.mean_ttft_ms, metrics.std_ttft_ms, 
                        metrics.median_ttft_ms, metrics.percentiles_ttft_ms)
    print_metric_details("Time per Output Token", metrics.mean_tpot_ms, metrics.std_tpot_ms,
                        metrics.median_tpot_ms, metrics.percentiles_tpot_ms)
    print_metric_details("Inter-token Latency", metrics.mean_itl_ms, metrics.std_itl_ms,
                        metrics.median_itl_ms, metrics.percentiles_itl_ms)
    print_metric_details("End-to-end Latency", metrics.mean_e2el_ms, metrics.std_e2el_ms,
                        metrics.median_e2el_ms, metrics.percentiles_e2el_ms)

    print("=" * 50)

    # 构建结果字典
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "mean_normalized_latency": metrics.mean_normalized_latency,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "mean_e2el_ms": metrics.mean_e2el_ms,
        "std_e2el_ms": metrics.std_e2el_ms,
        "median_e2el_ms": metrics.median_e2el_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": [output.output_tokens for output in outputs],
        "ttfts": [output.ttft for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    # 添加百分位数
    for p, value in metrics.percentiles_ttft_ms:
        result[f"p{int(p)}_ttft_ms"] = value
    for p, value in metrics.percentiles_tpot_ms:
        result[f"p{int(p)}_tpot_ms"] = value
    for p, value in metrics.percentiles_itl_ms:
        result[f"p{int(p)}_itl_ms"] = value
    for p, value in metrics.percentiles_e2el_ms:
        result[f"p{int(p)}_e2el_ms"] = value

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Orca serving throughput")
    
    # 基本参数
    parser.add_argument("--backend", type=str, default="orca", choices=list(ASYNC_REQUEST_FUNCS.keys()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")
    
    # 数据集参数
    parser.add_argument("--dataset-name", type=str, default="sharegpt", choices=["sharegpt"])
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the ShareGPT dataset")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process")
    parser.add_argument("--sharegpt-output-len", type=int, default=None, 
                       help="Override output length for ShareGPT dataset")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path")
    
    # 请求参数
    parser.add_argument("--request-rate", type=float, default=float("inf"), 
                       help="Request rate (req/s)")
    parser.add_argument("--burstiness", type=float, default=1.0,
                       help="Burstiness factor for request generation")
    
    # 输出参数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument("--metric-percentiles", type=str, default="99",
                       help="Comma-separated list of percentiles")

    args = parser.parse_args()
    
    print(f"Running benchmark with args: {args}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 构建API URL
    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    
    # 加载tokenizer
    tokenizer_id = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    # 加载数据集
    if args.dataset_name == "sharegpt":
        dataset = ShareGPTDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed
        )
        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
        )
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    # 解析百分位数
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    # 避免GC影响性能
    gc.collect()
    gc.freeze()

    # 运行基准测试
    benchmark_result = asyncio.run(
        benchmark(
            backend=args.backend,
            api_url=api_url,
            model_id=args.model,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            selected_percentiles=selected_percentiles,
        )
    )

    # 保存结果
    if args.save_result:
        if args.result_filename:
            filename = args.result_filename
        else:
            filename = get_timestamp_filename(args.backend, args.request_rate, args.model)
        
        # 添加运行配置到结果中
        benchmark_result.update({
            "backend": args.backend,
            "model": args.model,
            "dataset": args.dataset_name,
            "num_prompts": args.num_prompts,
            "request_rate": args.request_rate,
            "burstiness": args.burstiness,
            "timestamp": datetime.now().isoformat(),
        })
        
        write_to_json(benchmark_result, filename, args.result_dir)
        print(f"Results saved to: {filename}")


if __name__ == "__main__":
    main()