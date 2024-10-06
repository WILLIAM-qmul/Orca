import time
from scheduler.scheduler import Request
import pandas as pd
from scheduler.scheduler import OrcaScheduler
from engine_py.engine import ORCAExecutionEngine
import requests
import concurrent.futures

def load_requests_from_csv(file_path: str, column: str) -> list[Request]:
    df = pd.read_csv(file_path)
    requests = []

    for _, row in df.iterrows():
        prompt = str(row[column])
        requests.append(prompt)
    return requests

def send_request(prompt):
    url = "http://localhost:8000/generate"
    try:
        response = requests.post(url, json={"prompt": prompt})
        response.raise_for_status()
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

def simulate_user_requests(prompts: list[str], requests_per_second: int = 4, max_workers: int = 64) -> None:
    """Generate requests to the API of the scheduler using the list of given prompts.

    Args:
        requests (list[str]): list of prompts to be sent to the API taken from a file or a list of prompts.
        requests_per_second (int): specify the number of requests to be sent to the API per second.
    """
   
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(prompts), requests_per_second):
            batch = prompts[i:i+requests_per_second]
            for prompt in batch:
                executor.submit(send_request, prompt) 
            time.sleep(1)
    
def directly_process_requests(requests: list[str]) -> None:
    """Directly process user requests using the ORCAExecutionEngine without api calls in between the modules

    Args:
        requests (list[str]): list of prompts to be processed by the scheduler and Engine
    """
    engine = ORCAExecutionEngine()
    scheduler = OrcaScheduler(n_workers=4, engine=engine, max_batch_size=16, max_n_kv_slots=2000)
    scheduler.add_request_batch(requests[:50])
    scheduler.process_requests()
    

def main() -> None:
    requests: list[str] = load_requests_from_csv("../experiments/samples.csv", 'conversation')
    simulate_user_requests(requests, requests_per_second=4)

if __name__ == "__main__":
    main()
    