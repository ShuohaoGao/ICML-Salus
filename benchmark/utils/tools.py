from functools import partial
import json
import re
import multiprocessing
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Any, Iterable, Union, Tuple, Dict
from tqdm import tqdm


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_batch_infer_jsonl(messages_list: list) -> list:
    """
    Generate a batch of inference JSONL data.

    Args:
        messages_list (list): List of messages for inference.
        model_name (str): Name of the model.

    Returns:
        list: List of dictionaries containing the batch data, e.g.,
        {"custom_id": "request-1", "body": {"messages": [{"role": "user", "content": "天空为什么这么蓝？"}],"max_tokens": 16000}}
    """
    batch_data = []
    for i, messages in enumerate(messages_list):
        batch_data.append(
            json.dumps(
                {
                    "custom_id": f"request-{i+1}",
                    "body": {
                        "messages": messages,
                        "max_tokens": 16000,
                    },
                },
                ensure_ascii=False,
            )
        )
    return batch_data


def read_batch_results_from_jsonl(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = []
    for line in lines:
        data = json.loads(line)
        idx = int(data["custom_id"].split("-")[1])
        results.append(
            {
                "id": idx,
                "custom_id": data["custom_id"],
                "content": data["response"]["body"]["choices"][0]["message"]["content"],
            }
        )
    # 根据id排序
    results.sort(key=lambda x: x["id"])
    return results


def extract_json(s: str):
    """
    Extract JSON from a string.

    Args:
        s (str): The input string containing  ```json\nJSON-data\n```

    Returns:
        dict: The extracted JSON data.
    """
    # Use regex to find the JSON part in the string
    json_pattern = r"```json\n(.*?)\n```"
    match = re.search(json_pattern, s, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        print("No JSON found in the string.", s)
        return None
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", s)
        return None


def extract_label(s: str, label: str, strict: bool = True) -> str:
    """
    <label>content</label>
    -> content
    """
    end_tag = f"</{label}>"
    start_tag = f"<{label}>"
    end_pos = s.rfind(end_tag)
    if end_pos == -1:
        if strict:
            return None
        else:
            end_pos = len(s)
    start_pos = s.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None
    content = s[start_pos + len(start_tag) : end_pos].strip()
    return content
    # # pattern = Rf"<{label}>(.*?)</{label}>"
    # pattern = rf"<{label}>([\s\S]*?)<\/{label}>"
    # match = re.search(pattern, s)
    # if match:
    #     return match.group(1).strip()
    # else:
    #     return None


def _wrap_func(func: Callable, idx_args: tuple) -> tuple:
    """
    包装函数，执行原始函数并返回 (索引, 结果) 对。
    """
    idx, args = idx_args
    return (idx, func(*args))


def batch_process_parallel(
    func: Callable,
    args_list: List[Iterable],
    batch_size: int = 1,
    num_processes: int = None,
    use_tqdm: bool = False,
    use_thread: bool = False,
) -> List[Any]:
    """
    使用多进程或多线程并发地对一个函数进行批处理调用。

    这个函数会创建一个进程池，将任务列表分成多个批次（chunks），
    每个进程一次性处理一个批次的任务，从而提高效率，尤其是在任务数量很多时。
    结果会按照输入参数列表的顺序返回。

    Args:
        func (Callable): 需要并发执行的目标函数。
        args_list (List[Iterable]): 一个列表，其中每个元素都是一个可迭代对象（如列表或元组），
                                   包含了调用一次 func 时所需的参数。
                                   例如: 对于 func(a, b)，args_list 应为 [[a1, b1], [a2, b2], ...]。
        batch_size (int, optional): 每个进程一次处理的任务数量。默认为 1。
                                    增大 batch_size 可以减少进程间通信的开销，
                                    对于执行时间很短的函数特别有效。
        num_processes (int, optional): 使用的进程数量。默认为 None，表示使用 CPU 的核心数。
        use_thread (bool, optional): 是否使用多线程。默认为 False，表示使用多进程。

    Returns:
        List[Any]: 包含所有函数调用结果的列表，顺序与 args_list 对应。

    Raises:
        ValueError: 如果 batch_size 小于 1。

    Example:
        >>> def power(x, p):
        ...     return x ** p
        ...
        >>> args = [[i, 2] for i in range(10)]
        >>> results = batch_process_parallel(power, args, batch_size=3)
        >>> print(results)
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """
    if not args_list:
        return []

    if batch_size < 1:
        raise ValueError("batch_size 必须大于或等于 1")

    if not isinstance(args_list[0], (list, tuple)):
        args_list = [[arg] for arg in args_list]

    pool_cls = ThreadPool if use_thread else multiprocessing.Pool

    with pool_cls(processes=num_processes) as pool:
        if batch_size > 1 or not use_tqdm:
            # pool.starmap:
            # 1. func: 目标函数
            # 2. args_list: 参数列表的列表
            # 3. chunksize: 相当于 batch_size，它定义了多少个任务被打包送给一个工作进程
            # starmap 会自动解包每个子列表作为 func 的参数，例如 func(*[a1, b1])
            # 会保证返回结果的顺序与输入顺序一致
            results = pool.starmap(func, args_list, chunksize=batch_size)
        else:
            # 创建一个预先分配好空间的列表来存放结果
            results = [None] * len(args_list)
            # 1. 使用 enumerate(args_list) 直接生成 (index, args) 对。
            # 2. 使用 functools.partial 创建一个新函数
            #    这个新函数 'task_func' 只需要一个参数 (idx_args)
            #    因为它已经将 'func' 作为第一个参数绑定了
            task_func = partial(_wrap_func, func)
            # 3. 将新函数和 enumerate(args_list) 传递给 imap_unordered
            task_iterator = pool.imap_unordered(task_func, enumerate(args_list))

            pbar = tqdm(
                task_iterator,
                total=len(args_list),
                desc="Processing " + func.__name__,
                ncols=100,
            )
            for index, result in pbar:
                results[index] = result

    return results


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]


def write_jsonl(data: list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def analyze_distribution(
    data: List[int],
    percentiles: List[Union[int, float]] = [50, 75, 95, 99, 99.9, 100],
    thresholds: List[int] = [4096, 5120, 5632, 6144, 6656],
    ignore_zeros=True,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    对一个整数列表进行分布分析。

    该函数会执行两个操作：
    1. 计算指定百分位点（percentiles）对应的具体数值。
    2. 计算指定阈值（thresholds）在数据分布中所处的百分位。

    Args:
      data: 一个整数列表。
      percentiles: 需要计算其对应数值的百分位点列表。
      thresholds: 需要计算其所处百分位的数值阈值列表。

    Returns:
      一个元组，包含两个字典：
      - 第一个字典 (percentile_values): 映射“百分位”到“具体数值”。
        例如：{'99%': 5000}
      - 第二个字典 (threshold_percentiles): 映射“数值阈值”到其“所处的百分位”。
        例如：{'<=4096': '85.30%'}
    """
    import numpy as np

    # --- 输入验证 ---
    if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("输入的数据 'data' 必须是一个数值列表。")

    if not data:
        return {}, {}

    # --- 核心计算 ---
    # 将列表转换为NumPy数组以提高计算效率
    data = [x for x in data if x != 0]
    np_data = np.array(data)
    total_count = len(np_data)

    # 1. 计算指定百分位对应的数值
    percentile_values_raw = np.percentile(np_data, percentiles)
    percentile_values_result = {
        f"{p}%": v for p, v in zip(percentiles, percentile_values_raw)
    }
    for k, v in percentile_values_result.items():
        print(f"{k}: {int(v)}")

    # 2. 计算指定数值阈值在数据中所处的百分位
    threshold_percentiles_result = {}
    for t in thresholds:
        # 计算小于或等于当前阈值的元素数量
        count_le = np.sum(np_data <= t)
        # 计算该数量占总数的百分比
        percentile_of_threshold = (count_le / total_count) * 100
        # 将结果存入字典，格式化为保留两位小数的百分比字符串
        threshold_percentiles_result[f"<={t}"] = (
            f"{percentile_of_threshold:.2f}%, {total_count - count_le}"
        )
        print(
            f"<= {t}: {percentile_of_threshold:.2f}%   #rest: {total_count - count_le}"
        )

    print(f"avg: {sum(data) / len(data):.2f}  min: {min(data)}  max: {max(data)}")
    return percentile_values_result, threshold_percentiles_result
