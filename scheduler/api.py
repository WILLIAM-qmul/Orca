from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from scheduler.scheduler import OrcaScheduler
from models.request import Prompt_Request
from threading import Thread

app = FastAPI()

scheduler = OrcaScheduler(n_workers=4, max_batch_size=4, max_n_kv_slots=10**5)

@app.on_event("startup")
def start_background_tasks():
    Thread(target=scheduler.schedule_requests, daemon=True).start()
    
@app.post("/generate")
def process_request(request: Prompt_Request):
    request_id = scheduler.add_request(prompt=request.prompt)
    # should return once the request is completed:
    try:
        response = scheduler.get_completed_request(request_id).response
        scheduler.delete_request(request_id)
        return {"response": response, "status_code": 200}
    except Exception as e:
        print(f"request with request id {request_id} got lost: {e}")
        return {"response": "Error processing request", "status_code": 500}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)