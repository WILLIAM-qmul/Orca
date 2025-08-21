# orca

本项目基于论文 ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/system/files/osdi22-yu.pdf) 实现了分布式大模型推理系统，并针对 Llama 模型新增了适配的 `llama_engine` 和 `llama_decoder`，支持选择性批处理。

---

## 主要更新

- **engine_py 目录下新增适配 Llama 模型的 `llama_engine.py` 和 `llama_decoder.py`，支持高效批量推理。**
- **推荐使用 Dockerfile 进行环境安装，确保依赖一致。**
- **运行步骤详见 [steps.md](steps.md)。**
- **新增 benchmark 测试代码（见 `/benchmark` 目录），包含四个文件：**
  - `orca_benchmark_dataset.py`
  - `orca_benchmark_request_func.py`
  - `orca_benchmark_serving.py`
  - `orca_benchmark_utils.py`

---

## 安装与环境

建议直接使用项目自带的 Dockerfile 构建环境：

```bash
docker build -t orca -f docker/Dockerfile .
docker run -it --name orca_env -v /home/lsl/wwg/Orca:/home/lsl/wwg/Orca orca
```

如需本地安装，详见 [steps.md](steps.md) 的“安装依赖”部分。

---

## 运行步骤

详细步骤请参考 [steps.md](steps.md)，主要流程如下：

1. **先启动推理引擎（engine，监听 8080 端口）**
2. **再启动调度器（scheduler，监听 8000 端口）**
3. **最后用 benchmark 目录下的测试脚本或 test.py 进行请求测试**

---

## Benchmark 测试

`/benchmark` 目录下包含完整的基准测试工具，支持对 Orca/Llama 服务进行吞吐量和延迟评测。主要文件说明：

- `orca_benchmark_dataset.py`：数据集采样与处理
- `orca_benchmark_request_func.py`：后端请求函数
- `orca_benchmark_serving.py`：主测试入口，命令行参数丰富
- `orca_benchmark_utils.py`：通用工具函数

运行示例：

```bash
python benchmark/orca_benchmark_serving.py \
  --backend orca \
  --model Llama-2-7b-hf \
  --dataset-name sharegpt \
  --dataset-path <path_to_sharegpt.json> \
  --request-rate 10 \
  --num-prompts 100
```

---

## 架构图

![orca-arch](https://github.com/user-attachments/assets/033df6bd-a3d5-43a8-bd97-6b464e594028)

---

如需详细了解各模块结构和开发流程，请参考 [architecture.md](architecture.md) 和 [steps.md](steps.md)。