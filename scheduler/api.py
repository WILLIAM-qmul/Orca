from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from scheduler.scheduler import OrcaScheduler # 导入调度器核心类
from models.request import Prompt_Request # 导入单条请求的数据结构
from threading import Thread

app = FastAPI() # 创建FastAPI应用实例

# 实例化调度器，设置4个worker，最大batch为4，最大KV缓存槽为10万
# scheduler = OrcaScheduler(n_workers=4, max_batch_size=4, max_n_kv_slots=10**5)
scheduler = OrcaScheduler(n_workers=1, max_batch_size=1, max_n_kv_slots=10**5)

@app.on_event("startup")
def start_background_tasks():
    # 启动调度器的主循环线程，daemon=True表示主进程退出时自动关闭
    Thread(target=scheduler.schedule_requests, daemon=True).start()
    
@app.post("/generate") # 定义POST接口/generate，处理推理请求
def process_request(request: Prompt_Request):
    # 调用调度器添加请求，返回请求ID
    request_id = scheduler.add_request(prompt=request.prompt)
    # print(f"Added request with request prompt: {request.prompt}")
    # should return once the request is completed:
    # 等待该请求完成后返回结果
    try:
        # print(f"waiting for request with request id {request_id} to complete")
        response = scheduler.get_completed_request(request_id).response
        # print(f"request with request id {request_id} completed")
        scheduler.delete_request(request_id)
        return {"response": response, "status_code": 200}
    except Exception as e:
        # print(f"request with request id {request_id} got lost: {e}")
        return {"response": "Error processing request", "status_code": 500}


if __name__ == "__main__":
    # 启动FastAPI服务，监听0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)