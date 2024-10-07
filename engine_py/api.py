from fastapi import FastAPI
import uvicorn
from models.request import Batch, Batch_Response, Batch_Response_Item
from .engine import ORCAExecutionEngine
from .llm import LLM
import json

app = FastAPI()

engine = ORCAExecutionEngine()
llm = LLM(model="facebook/opt-125m", device="cpu")


@app.post("/process_batch")
def process_batch(batch: Batch) -> Batch_Response:
    prompts = [request.prompt for request in batch.requests]
    responses = llm.batch_process(prompts=prompts)
    if len(responses) == len(batch.requests):
        batch_response = [Batch_Response_Item(request_id=request.request_id, generated_tokens=response, request_completed=True) for request, response in zip(batch.requests, responses)]
        return {"responses": batch_response, "status_code": 200}
    else:
        return {"responses": batch_response,  "message": "Error processing some of the requests", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)