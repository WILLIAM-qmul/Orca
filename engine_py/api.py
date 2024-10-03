from fastapi import FastAPI
import uvicorn
from models.request import Batch
from .engine import ORCAExecutionEngine
from .llm import LLM
import json

app = FastAPI()

engine = ORCAExecutionEngine()
llm = LLM(model="facebook/opt-125m", device="cpu")


@app.post("/process_iteration_batch")
def process_iteration_batch(batch: Batch):
    # Process the request here
    # responses = engine.execute(requests)
    prompts = [request.prompt for request in batch.requests]
    responses = llm.batch_process(prompts=prompts)
    if len(responses) == len(batch.requests):
        return {"responses": responses, "status_code": 200}
    else:
        return {"responses": responses,  "message": "Error processing some of the requests", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)