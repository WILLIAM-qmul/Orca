import time
from scheduler.scheduler import Request
import pandas as pd
from scheduler.scheduler import OrcaScheduler
from engine_py.engine import ORCAExecutionEngine
import requests # 导入requests库，用于发送HTTP请求
import concurrent.futures # 导入并发库，用于多线程发送请求

def load_requests_from_csv(file_path: str, column: str) -> list[Request]:
    # 从csv文件读取指定列的内容，返回请求列表
    df = pd.read_csv(file_path) # 读取csv文件为DataFrame
    requests = [] # 初始化请求列表

    for _, row in df.iterrows(): # 遍历每一行
        prompt = str(row[column]) # 获取指定列内容并转为字符串
        requests.append(prompt) # 添加到请求列表
    return requests # 返回请求列表

def send_request(prompt):
    # 向调度器API发送单条请求
    url = "http://localhost:8000/generate" # 调度器API地址
    try:
        response = requests.post(url, json={"prompt": prompt}) # 发送POST请求，传递prompt
        response.raise_for_status()
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

def simulate_user_requests(prompts: list[str], batch_size: int = 1, batch_freq: int = 1, max_workers: int = 64) -> None:
    """Generate requests to the API of the scheduler using the list of given prompts.

    Args:
        requests (list[str]): list of prompts to be sent to the API taken from a file or a list of prompts.
        batch_size (int): specify the number of requests to be sent to the API per batch_freqency.
        batch_freq (int): specify how often (in seconds) a batch should be sent to the API.
    """
    """
    使用多线程模拟用户批量请求调度器API。

    Args:
        requests (list[str]): 要发送的prompt列表
        batch_size (int): 每批发送的请求数
        batch_freq (int): 每批请求之间的间隔（秒）
        max_workers (int): 最大线程数
    """
   
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor: # 创建线程池
        for i in range(0, len(prompts), batch_size): # 按batch_size分批
            batch = prompts[i:i+batch_size] # 当前批次
            for prompt in batch:
                executor.submit(send_request, prompt) # 为每个prompt提交一个线程任务
                # 每个 batch（即 batch_size 个 prompt）会用线程池为每个 prompt 启动一个线程，这些线程会并发（同时）发送请求。
            time.sleep(batch_freq) # 批次间隔
    
def directly_process_requests(requests: list[str]) -> None:
    """Directly process user requests using the ORCAExecutionEngine without api calls in between the modules

    Args:
        requests (list[str]): list of prompts to be processed by the scheduler and Engine
    """
    """
    直接用本地Engine和Scheduler处理请求（不通过API），用于本地测试。

    Args:
        requests (list[str]): 要处理的prompt列表
    """
    engine = ORCAExecutionEngine()
    scheduler = OrcaScheduler(n_workers=4, engine=engine, max_batch_size=16, max_n_kv_slots=2000)
    scheduler.add_request_batch(requests[:50])
    scheduler.process_requests()
    

def main() -> None:
    # 主函数，加载csv并模拟用户请求
    requests: list[str] = load_requests_from_csv("/home/lsl/wwg/orca/experiments/samples.csv", 'conversation')
    simulate_user_requests(requests,  batch_size=1, batch_freq=3) # 逐条、每3秒发送一次请求

if __name__ == "__main__":
    main() # 脚本入口，执行主函数
    