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
    """
    OrcaScheduler类负责管理请求池，并将请求批量发送到推理引擎处理。

    属性说明：
        ENGINE_URL (str): 推理引擎的HTTP接口地址
        request_pool (dict[int, Request]): 所有待处理请求的池
        engine (ORCAExecutionEngine): 推理引擎实例（可选）
        n_workers (int): 并发worker数量
        MAX_BATCH_SIZE (int): 每批最大请求数
        max_n_kv_slots (int): 最大KV缓存槽数
        n_kv_slots_rsrvd (int): 当前已预留的KV槽数
        current_request_id (int): 当前最新分配的请求ID
        request_id_lock (threading.Lock): 请求ID自增锁
        request_lock (threading.Lock): 请求池操作锁
    """
    ENGINE_URL: str = "http://0.0.0.0:8080/process_batch" # 推理引擎的HTTP接口地址
    
    def __init__(self, engine: Optional[ORCAExecutionEngine] = None, n_workers: int = 4, max_batch_size: int = 16, max_n_kv_slots: int = 10**5) -> None:
        self.request_pool: dict[int, Request] = {} # 请求池，key为request_id，value为Request对象
        self.n_workers = n_workers
        self.engine = engine # 推理引擎实例
        self.MAX_BATCH_SIZE = max_batch_size # 每批最大请求数
        self.max_n_kv_slots = max_n_kv_slots # 最大KV缓存槽数
        self.n_kv_slots_rsrvd = 0 # 当前已预留的KV槽数
        self.current_request_id = 0 # 当前最新分配的请求ID
        self.request_id_lock = threading.Lock()
        self.request_lock = threading.Lock()
        
    def increment_request_id(self):
        # 原子操作，自增请求ID
        with self.request_id_lock:
            self.current_request_id += 1
            return self.current_request_id
        
    def calculate_max_tokens(self, prompt: str) -> int:
        '''Calculate the maximum tokens required for a given prompt including all potential output tokens.'''
        '''计算该请求最大token数（输入词数+预留输出token数）'''
        prompt_tokens = len(prompt.split()) # Use word count as max_tokens for simplicity (change later)  # 以空格分词计数（可根据实际模型调整）
        reserved_output_tokens = 1024 # 预留输出token数
        return prompt_tokens + reserved_output_tokens
    # def calculate_max_tokens(self, prompt: str) -> int:
    #     '''Calculate the maximum tokens required for a given prompt including all potential output tokens.'''
    #     # Use proper tokenizer for accurate token counting
    #     prompt_tokens = len(self.tokenizer.encode(prompt))
    #     reserved_output_tokens = 1024
    #     return prompt_tokens + reserved_output_tokens
        

    def add_request(self, prompt: str):
        """添加新请求到请求池，并返回请求ID"""
        """Add a request to the request pool and returns the request_id.

        Args:
            prompt (str): prompt to be processed

        Returns:
            int: request_id of added request
        """
        with self.request_lock:
            request_id = self.increment_request_id() # 获取新请求ID
            max_tokens = self.calculate_max_tokens(prompt) # 计算最大token数
            new_request = Request(prompt=prompt, max_tokens=max_tokens, request_id=request_id, state=RequestState.INITIATION) # 创建Request对象
            self.request_pool[request_id] = new_request # 加入请求池
            
            # print(f"Request {self.current_request_id} submitted: {prompt[:30]}...") # 打印前30字符
            return request_id # 返回请求ID
            
    def select(self) -> dict[Request]:
        """从请求池中选择一批可处理的请求，返回字典（request_id: Request）"""
        """Select a batch of requests to process based on the current request pool and the number of reserved slots."""
        batch: dict[Request] = {} # 当前批次
        request_pool = [req for req in self.request_pool.values() if req.state != RequestState.RUNNING and req.state != RequestState.COMPLETED] # 只选取未运行和未完成的请求
        print(f"Selecting batch from {len(request_pool)} requests")
        request_pool.sort(key=lambda x: x.request_id) # 按请求ID排序（先进先出）

        for req in request_pool:
            if len(batch) == self.MAX_BATCH_SIZE: # 达到最大batch则停止
                break
            if req.state == RequestState.INITIATION:
                new_n_rsrv = self.n_kv_slots_rsrvd + req.max_tokens # 计算新预留槽数
                print(f"New n_rsrv: {new_n_rsrv}, Max n_kv_slots: {self.max_n_kv_slots}")
                if new_n_rsrv > self.max_n_kv_slots:  # 超出最大槽数则停止
                    break
                self.n_kv_slots_rsrvd = new_n_rsrv # 更新已预留槽数
            batch[req.request_id] = req # 加入当前批次
            
        return batch # 返回批次
    
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
        """
        将一批请求发送到推理引擎处理，返回响应结果
        """

        print(f"Scheduling batch of {len(batch.values())} requests")
        req_list = []
        for req in batch.values():
            with self.request_lock:
                req_list.append(Batch_Item(prompt=req.prompt, request_id=req.request_id)) # 构造Batch_Item
                req.state = RequestState.RUNNING # 标记为运行中
        engine_batch = Batch(requests=req_list) # 构造Batch对象
        response = requests.post(self.ENGINE_URL, json=engine_batch.model_dump()) # 发送POST请求到推理引擎
        response.raise_for_status()
        print(f"Received response from engine: {response.json()}")
        
        return response.json()
                
    def process_batch_response(self, future: Future, batch: dict[int, Request]):
        """处理推理引擎返回的批量响应，更新请求池状态"""
        """Process the response from the execution engine for a batch of requests.

        Args:
            future (Future): Future thread object representing the response from the execution engine.
        """
        try:
            response = future.result() # 获取线程执行结果
            print(f"Processing response: {response}")
            response = Batch_Response.model_validate(response) # 校验并转为Batch_Response对象
            print(f"{len(response.responses)} responses received.")
            for response_item in response.responses:
                with self.request_lock:
                    req = self.request_pool.get(response_item.request_id) # 获取对应请求
                    req.response += response_item.generated_tokens # 累加生成文本
                    req.prompt += response_item.generated_tokens # 累加到prompt（用于增量生成）
                    print(f"Request {response_item.request_id} response: {response_item.generated_tokens}, total request: {req.response}")
                    if response_item.request_completed: # 如果已完成
                        self.n_kv_slots_rsrvd -= req.max_tokens # 释放已预留槽数
                        print(f"Request {response_item.request_id} completed.")
                        req.mark_as_completed() # 标记为完成
                    else:
                        req.state = RequestState.INCREMENT
        except Exception as e:
            print(f"Error processing batch: {e}")

    
    def schedule_requests(self) -> None:
        """Schedule the requests in the request pool using the ORCA scheduling algorithm."""
        """主调度循环，不断从请求池中取batch并分发到推理引擎"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            n_scheduled = 0 # 当前已调度的任务数
            futures = {} # To store threads and their corresponding batches # 存储future和对应batch

            while True:
                batch = self.select() # 选取一批请求

                if batch:
                    futures[executor.submit(self.send_batch_to_engine, batch)] = batch # 提交到线程池
                    n_scheduled += 1
                
                # If all worker threads are engaged, wait for any thread to complete
                # 如果所有worker都在忙，等待有任务完成
                while n_scheduled >= self.n_workers:
                    completed_futures, _ = wait(futures.keys(), return_when='FIRST_COMPLETED')
                    for future in list(completed_futures):
                        self.process_batch_response(future=future, batch=futures[future])
                        del futures[future]
                        n_scheduled -= 1
                
                # 检查并处理已完成的任务
                for future in list(futures):
                    if future.done():
                        self.process_batch_response(future=future, batch=futures[future])
                        del futures[future]
                        n_scheduled -= 1

                # 可选：打印剩余请求数
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
        """阻塞等待指定请求完成，返回Request对象"""
        with self.request_lock:
            req = self.request_pool.get(request_id)
            if req is None:
                raise ValueError(f"Request with id {request_id} not found.")
            if req.state == RequestState.COMPLETED:
                return req
        req.wait_for_completion() # 阻塞等待完成
        return req
    
    def delete_request(self, request_id: int) -> None:
        """从请求池中删除指定请求"""
        """Remove a request from the request pool.

        Args:
            request_id (int): The id of the request to remove.
        """
        with self.request_lock:
            del self.request_pool[request_id]
            
