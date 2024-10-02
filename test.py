from scheduler import Request
import pandas as pd
from scheduler import Scheduler
from engine import ORCAExecutionEngine

def load_requests_from_csv(file_path: str, column: str) -> list[Request]:
    df = pd.read_csv(file_path)
    requests = []

    for _, row in df.iterrows():
        prompt = str(row[column])
        requests.append(prompt)
    return requests

def main() -> None:
    requests: list[str] = load_requests_from_csv("../experiments/samples.csv", 'conversation')
    engine = ORCAExecutionEngine()
    scheduler = Scheduler(n_workers=4, engine=engine, max_batch_size=16, n_kv_slots=2000)
    scheduler.add_request_batch(requests[:50])
    scheduler.process_requests()
    


if __name__ == "__main__":
    main()
    