"""
运行6.2实验的批量测试脚本
"""
import os


def run_benchmark():
    """运行Orca基准测试"""
    # 定义参数
    backend_list = ["orca"]
    # request_rates = [0.2, 0.4, 0.6, 0.8, 1, 3, 5, 7, 9]
    request_rates = [1]
    dataset_name = "sharegpt"
    dataset_path = "/home/lsl/wwg/Orca/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
    model_name = "/home/lsl/wwg/Orca/Llama-2-7b-hf"
    # tokenizer = "/home/lsl/wwg/Orca/Llama-2-7b-hf"
    endpoint = "/generate"
    result_dir = "6.2_results"
    num_prompts = 1000

    # 创建结果文件夹
    os.makedirs(result_dir, exist_ok=True)

    # 循环运行不同后端和请求速率
    for backend in backend_list:
        for rate in request_rates:
            # 构造文件名
            file_name = f"{result_dir}/{backend}_rate{rate}_on_dataset_{dataset_name}_with_{model_name}.json"

            # 构造命令
            command = f"""
            python /home/lsl/wwg/Orca/benchmark/orca_benchmark_serving.py \
                --backend {backend} \
                --model {model_name} \
                --endpoint {endpoint} \
                --dataset-name {dataset_name} \
                --dataset-path {dataset_path} \
                --request-rate {rate} \
                --num-prompts {num_prompts} \
                --save-result \
                --result-filename {file_name} \
                --seed 0
            """

            # 打印命令并执行
            print(f"Running: {command}")
            exit_code = os.system(command)
            
            if exit_code != 0:
                print(f"Command failed with exit code {exit_code}")
            else:
                print(f"Completed: {backend} at rate {rate}")


if __name__ == "__main__":
    run_benchmark()