from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from scheduler.scheduler import OrcaScheduler
from models.request import Prompt_Request
from threading import Thread

app = FastAPI()

scheduler = OrcaScheduler(n_workers=4, max_batch_size=4, max_n_kv_slots=2000)

@app.on_event("startup")
def start_background_tasks():
    Thread(target=scheduler.schedule_requests, daemon=True).start()
    
@app.post("/generate")
def process_request(request: Prompt_Request):
    request_id = scheduler.add_request(prompt=request.prompt)
    # should return once the request is completed:


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)