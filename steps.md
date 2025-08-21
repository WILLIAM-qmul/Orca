项目是一个基于 FastAPI 的分布式推理系统，包含调度器（scheduler）和推理引擎（engine），需要分别启动两个服务。下面是详细的运行步骤：

---

## 1. 安装依赖
docker build -t orca -f docker/Dockerfile .
docker run -it --name orca_env orca
or
docker run -it --name orca_env \
  -v /home/lsl/wwg/Orca:/home/lsl/wwg/Orca \
  orca

docker rm orca_env
docker start -ai orca_env

docker exec -it orca_env bash

确保你已安装 Python 3.8+ 和 pip。进入项目根目录，安装依赖（建议使用虚拟环境）：

```bash
cd /home/lsl/wwg/orca
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

如果没有 requirements.txt，你需要手动安装主要依赖：

```bash
pip install fastapi uvicorn transformers torch requests pandas
```

---

## 2. 启动推理引擎服务（engine）

推理引擎监听 8080 端口，需先启动：

```bash
cd /home/lsl/wwg/orca
uvicorn engine_py.api:app --host 0.0.0.0 --port 8080
```

保持该终端窗口开启。

---

## 3. 启动调度器服务（scheduler）

调度器监听 8000 端口，负责接收用户请求并调度到推理引擎：

```bash
cd /home/lsl/wwg/orca
uvicorn scheduler.api:app --host 0.0.0.0 --port 8000
```

或者你可以直接用 VS Code 的调试功能，选择 launch.json 里的 "FastAPI as Module" 配置启动。

---

## 4. 发送请求测试

你可以用 test.py 脚本模拟用户请求：

```bash
python test.py
```

或者用 curl 手动测试：

```bash
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "你好，介绍一下你自己"}'
```

---

## 5. 典型开发流程

1. **先启动 engine（8080）**
2. **再启动 scheduler（8000）**
3. **最后用 test.py 或 curl 发送请求**

---

## 6. 常见问题排查

- **端口冲突**：确保 8000 和 8080 没有被其他进程占用。
- **依赖缺失**：如遇 ImportError，检查 requirements 是否安装齐全。
- **模型下载慢**：首次运行 transformers 相关代码会自动下载模型，需耐心等待。

---

如需停止服务，按 `Ctrl+C` 即可。

如有具体报错，可贴出错误信息进一步排查。