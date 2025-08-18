# from fastapi import FastAPI
# import uvicorn
# from models.request import Batch, Batch_Response, Batch_Response_Item
# from .engine import ORCAExecutionEngine
# from .llm import LLM
# import json
# from .opt_engine import OPT_Engine

# app = FastAPI()

# engine = ORCAExecutionEngine()
# opt_engine = OPT_Engine()
# llm = LLM(model="facebook/opt-125m", device="cpu")


# @app.post("/process_batch")
# def process_batch(batch: Batch) -> Batch_Response:
#     #responses = llm.batch_process(prompts=prompts)
#     print(f"Received batch: {batch.requests}")
#     responses = opt_engine.batch_process(requests=batch.requests, max_generation_length=1)

#     if len(responses) == len(batch.requests):
#         batch_response = [Batch_Response_Item(request_id=request.request_id, generated_tokens=text, request_completed=completed) for request, (text, completed) in zip(batch.requests, responses)]
#         return Batch_Response(responses=batch_response)
#     else:
#         return {"responses": Batch_Response(responses=[]),  "message": "Error processing some of the requests", "status_code": 500}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

from fastapi import FastAPI
import uvicorn
from models.request import Batch, Batch_Response, Batch_Response_Item
from .llama_engine import Llama_Engine

app = FastAPI()

# 初始化 Llama 引擎
llama_engine = Llama_Engine(model_path="/home/lsl/wwg/models/Llama-2-7b-hf")

@app.post("/process_batch")
def process_batch(batch: Batch) -> Batch_Response:
    # print(f"接收到批量请求: {batch.requests}")
    responses = llama_engine.batch_process(requests=batch.requests, max_generation_length=1)

    if len(responses) == len(batch.requests):
        batch_response = [
            Batch_Response_Item(
                request_id=request.request_id, 
                generated_tokens=text, 
                request_completed=completed
            ) 
            for request, (text, completed) in zip(batch.requests, responses)
        ]
        return Batch_Response(responses=batch_response)
    else:
        return {"responses": Batch_Response(responses=[]), 
                "message": "处理部分请求时出错", 
                "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)