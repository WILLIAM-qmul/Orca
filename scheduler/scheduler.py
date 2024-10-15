import time
from typing import Optional
from engine_py.engine import ORCAExecutionEngine
from threading import Thread
from models.request import Batch, Batch_Item, Batch_Response, Batch_Response_Item, Request, RequestState
import threading
import requests
from concurrent.futures import Future, ThreadPoolExecutor, wait
import json


class OrcaScheduler:
    """
    The Scheduler class manages the request pool and send requests to the execution engine for processing.
    
    Attributes:
        ENGINE_URL (str): The URL of the execution engine to send the requests to.
        request_pool (list[Request]): A list of requests to be processed.
        engine (ORCAExecutionEngine): The execution engine to process the requests.
        n_workers (int): The number of workers to process the requests.
        max_batch_size (int): The maximum number of requests to be processed in a batch (called max_bs in the Orca paper).
        max_n_kv_slots (int): The maximum number of key-value slots available for processing (called n_slots in the Orca paper).
        n_kv_slots_rsrvd (int): The number of key-value slots reserved for processing the requests in the current batch  (called n_rsrv in the Orca paper).
        current_request_id (int): The current request id that was assigned to the most recently added request.
        request_id_lock (threading.Lock): A lock to ensure the request id is incremented atomically.
    """
    ENGINE_URL: str = "http://0.0.0.0:8080/process_batch"
    
    def __init__(self, engine: Optional[ORCAExecutionEngine] = None, n_workers: int = 4, max_batch_size: int = 16, max_n_kv_slots: int = 10**5) -> None:
        self.request_pool: dict[int, Request] = {}
        self.n_workers = n_workers
        self.engine = engine
        self.MAX_BATCH_SIZE = max_batch_size
        self.max_n_kv_slots = max_n_kv_slots
        self.n_kv_slots_rsrvd = 0
        self.current_request_id = 0
        self.request_id_lock = threading.Lock()
        self.request_lock = threading.Lock()
        
    def increment_request_id(self):
        with self.request_id_lock:
            self.current_request_id += 1
            return self.current_request_id
        
    def calculate_max_tokens(self, prompt: str) -> int:
        '''Calculate the maximum tokens required for a given prompt including all potential output tokens.'''
        prompt_tokens = len(prompt.split()) # Use word count as max_tokens for simplicity (change later)
        reserved_output_tokens = 1024
        return prompt_tokens + reserved_output_tokens
        

    def add_request(self, prompt: str):
        """Add a request to the request pool and returns the request_id.

        Args:
            prompt (str): prompt to be processed

        Returns:
            int: request_id of added request
        """
        with self.request_lock:
            request_id = self.increment_request_id()
            max_tokens = self.calculate_max_tokens(prompt)
            new_request = Request(prompt=prompt, max_tokens=max_tokens, request_id=request_id, state=RequestState.INITIATION)
            self.request_pool[request_id] = new_request
            
            print(f"Request {self.current_request_id} submitted: {prompt[:30]}...")
            return request_id
            
    def select(self) -> dict[Request]:
        """Select a batch of requests to process based on the current request pool and the number of reserved slots."""
        batch: dict[Request] = {}
        request_pool = [req for req in self.request_pool.values() if req.state != RequestState.RUNNING and req.state != RequestState.COMPLETED]
        print(f"Selecting batch from {len(request_pool)} requests")
        request_pool.sort(key=lambda x: x.request_id)

        for req in request_pool:
            if len(batch) == self.MAX_BATCH_SIZE:
                break
            if req.state == RequestState.INITIATION:
                new_n_rsrv = self.n_kv_slots_rsrvd + req.max_tokens
                print(f"New n_rsrv: {new_n_rsrv}, Max n_kv_slots: {self.max_n_kv_slots}")
                if new_n_rsrv > self.max_n_kv_slots: 
                    break
                self.n_kv_slots_rsrvd = new_n_rsrv
            batch[req.request_id] = req
            
        return batch
    
    def send_batch_to_engine(self, batch: dict[int, Request]):
        """
        Send a batch of requests to the execution engine for processing.
        
        Args:
            batch (list[Request]): A batch of requests to be processed.
        Returns:
            dict: The response from the execution engine.
        Raises:
            requests.exceptions.HTTPError: If the request to the execution engine fails.
        """

        print(f"Scheduling batch of {len(batch.values())} requests")
        req_list = []
        for req in batch.values():
            with self.request_lock:
                req_list.append(Batch_Item(prompt=req.prompt, request_id=req.request_id))
                req.state = RequestState.RUNNING
        engine_batch = Batch(requests=req_list)
        response = requests.post(self.ENGINE_URL, json=engine_batch.model_dump())
        response.raise_for_status()
        print(f"Received response from engine: {response.json()}")
        
        return response.json()
                
    def process_batch_response(self, future: Future, batch: dict[int, Request]):
        """Process the response from the execution engine for a batch of requests.

        Args:
            future (Future): Future thread object representing the response from the execution engine.
        """
        try:
            response = future.result()
            print(f"Processing response: {response}")
            response = Batch_Response.model_validate(response)
            print(f"{len(response.responses)} responses received.")
            for response_item in response.responses:
                with self.request_lock:
                    req = self.request_pool.get(response_item.request_id)
                    req.response += response_item.generated_tokens
                    req.prompt += response_item.generated_tokens
                    print(f"Request {response_item.request_id} response: {response_item.generated_tokens}, total request: {req.response}")
                    if response_item.request_completed:
                        self.n_kv_slots_rsrvd -= req.max_tokens
                        print(f"Request {response_item.request_id} completed.")
                        req.mark_as_completed()
                    else:
                        req.state = RequestState.INCREMENT
        except Exception as e:
            print(f"Error processing batch: {e}")

    
    def schedule_requests(self) -> None:
        """Schedule the requests in the request pool using the ORCA scheduling algorithm."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            n_scheduled = 0
            futures = {} # To store threads and their corresponding batches

            while True:
                batch = self.select()

                if batch:
                    futures[executor.submit(self.send_batch_to_engine, batch)] = batch
                    n_scheduled += 1
                
                # If all worker threads are engaged, wait for any thread to complete
                while n_scheduled >= self.n_workers:
                    completed_futures, _ = wait(futures.keys(), return_when='FIRST_COMPLETED')
                    for future in list(completed_futures):
                        self.process_batch_response(future=future, batch=futures[future])
                        del futures[future]
                        n_scheduled -= 1
                
                for future in list(futures):
                    if future.done():
                        self.process_batch_response(future=future, batch=futures[future])
                        del futures[future]
                        n_scheduled -= 1

                #print(f"Requests Yet To Be Processed: {len(self.request_pool)}")
                
    def get_completed_request(self, request_id: int) -> Request:
        """Get the completed request from the request pool once it completes.

        Args:
            request_id (int): The id of the request to get.

        Returns:
            Request: The completed request if found in the request pool.
        Raises:
            ValueError: If the request with the specified id is not found in the request pool.
        """
        with self.request_lock:
            req = self.request_pool.get(request_id)
            if req is None:
                raise ValueError(f"Request with id {request_id} not found.")
            if req.state == RequestState.COMPLETED:
                return req
        req.wait_for_completion()
        return req
    
    def delete_request(self, request_id: int) -> None:
        """Remove a request from the request pool.

        Args:
            request_id (int): The id of the request to remove.
        """
        with self.request_lock:
            del self.request_pool[request_id]
            
