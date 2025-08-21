"""
基准测试工具函数
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List


def write_to_json(
    data: Dict[str, Any], 
    filename: str, 
    result_dir: str = None
) -> None:
    """将数据写入JSON文件"""
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, filename)
    else:
        filepath = filename

    # 新增：自动创建父目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_to_pytorch_benchmark_format(
    args,
    metrics: Dict[str, List[float]],
    extra_info: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """转换为PyTorch基准测试格式"""
    records = []
    
    for metric_name, values in metrics.items():
        record = {
            "name": metric_name,
            "backend": args.backend,
            "model": args.model,
            "request_rate": args.request_rate,
            "num_prompts": args.num_prompts,
            "timestamp": datetime.now().isoformat(),
            "value": values[0] if values else 0,
        }
        
        if extra_info:
            record.update(extra_info)
            
        records.append(record)
    
    return records


def calculate_percentiles(values: List[float], percentiles: List[float]) -> List[tuple]:
    """计算百分位数"""
    import numpy as np
    if not values:
        return [(p, 0.0) for p in percentiles]
    return [(p, np.percentile(values, p)) for p in percentiles]


def get_timestamp_filename(backend: str, request_rate: float, model_name: str) -> str:
    """生成带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    return f"{backend}_{request_rate}qps_{safe_model_name}_{timestamp}.json"