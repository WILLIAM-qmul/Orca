from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from scheduler import Scheduler
import pandas as pd
from models.request import Prompt_Request, Batch_Prompt_Request, Scheduled_Iteration, Iteration_Responses
from orca.engine_py.engine import ORCAExecutionEngine



app = FastAPI()

engine = ORCAExecutionEngine()


@app.post("/process_iteration")
def process_request(requests: Scheduled_Iteration):
    # Process the request here
    responses = engine.execute(requests)
    if len(responses) == len(requests):
        return {"responses": responses, "status_code": 200}
    else:
        return {"responses": responses,  "message": "Error processing some of the requests", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)