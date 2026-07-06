import re
import time
from openai import OpenAI
import requests
import json
import datetime
import sys
import random
from .tools import extract_label
from .io_func import read_json, write_json
import os
import subprocess


# 从本地配置文件加载 API_key
_local_config_path = os.path.join(os.path.dirname(__file__), "API_key.json")
if os.path.exists(_local_config_path):
    with open(_local_config_path, "r") as _f:
        online_LLM_dict = json.load(_f)
else:
    print(
        f"[WARNING] 未找到本地LLM配置文件: {_local_config_path}，请参考 API_key_example.json 创建"
    )
    online_LLM_dict = {
        "Qwen3-32B": {
            "API_key": "xxxxxxxxxxxxxxx",
            "base_URL": "http://localhost:8011/v1",
        },
        "Qwen3-8B": {
            "API_key": "xxxxxxxxxxxxxxx",
            "base_URL": "http://localhost:8000/v1",
        },
    }



# online api llm
default_online_generation_kwargs = {
    "max_tokens": 8192,
}

thinking_token_ending = "</think>\n\n"  # considering Local LLM


def response_from_online_LLM(
    messages,
    model_name="GLM-4-Flash",
    API_key=None,
    base_URL=None,
    generation_kwargs={},
    split_reason_content=False,
):
    """
    :param: split_reason_content
                if Ture, return (answer, reason_text)
                if False, return {reason_text}{thinking_token_ending}{ans}
    """
    origin_model_name = model_name
    response_text = ""
    if model_name.endswith("-enable_thinking"):
        # 用于处理Deepseek-V3.1 3.2等混合思考模型
        if not generation_kwargs:
            generation_kwargs = default_online_generation_kwargs.copy()
        generation_kwargs.update(
            {"extra_body": {"thinking": {"type": "enabled"}}}
        )  #  包含思维链
        model_name = model_name[: -len("-enable_thinking")]
    elif model_name.endswith("-disable_thinking"):
        if not generation_kwargs:
            generation_kwargs = default_online_generation_kwargs.copy()
        if model_name.startswith("deepseek-v4-"):
            generation_kwargs.update({"extra_body": {"thinking": {"type": "disabled"}}})
        else:
            generation_kwargs.update(
                {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            )
        model_name = model_name[: -len("-disable_thinking")]
    elif model_name.endswith("-enable_reasoning"):
        # 用于处理GPT-5.2等混合思考模型
        if not generation_kwargs:
            generation_kwargs = default_online_generation_kwargs.copy()
        generation_kwargs.update(
            {"extra_body": {"reasoning": {"enabled": True}}}
        )  #  包含思维链
        model_name = model_name[: -len("-enable_reasoning")]
    
    if model_name in [
        "Pro/deepseek-ai/DeepSeek-V3.2",
        "Pro/zai-org/GLM-4.7",
    ]:
        # 硅基流动 API : RPM 为 30,000；TPM 为 5,000,000
        response_text = response_siliconflow_LLM(
            messages,
            model_name,
            enable_thinking=all(
                word in str(generation_kwargs) for word in ["thinking", "enabled"]
            ),
        )
    else:
        if API_key is None:
            API_key = online_LLM_dict[model_name]["API_key"]
        if base_URL is None:
            base_URL = online_LLM_dict[model_name]["base_URL"]
        client = OpenAI(
            api_key=API_key,
            base_url=base_URL,
            timeout=180.0,
        )
        if generation_kwargs is None:
            generation_kwargs = default_online_generation_kwargs.copy()
        chat_completion = client.chat.completions.create(
            messages=messages, model=model_name, **generation_kwargs
        )
        # print(chat_completion.usage.prompt_cache_hit_tokens, flush=True)
        ans = chat_completion.choices[0].message.content
        reason_text = ""
        if hasattr(chat_completion.choices[0].message, "reasoning_content"):
            reason_text = chat_completion.choices[0].message.reasoning_content
        elif hasattr(chat_completion.choices[0].message, "reasoning"):
            reason_text = chat_completion.choices[0].message.reasoning
        response_text = f"{reason_text}{thinking_token_ending}{ans}"
    if not split_reason_content:
        return response_text
    else:
        raw_response_text_parts = response_text.split(thinking_token_ending)
        answer, reason_text = (
            raw_response_text_parts[-1].strip(),
            raw_response_text_parts[0].strip(),
        )
        if model_name in ["Qwen3-32B"] and origin_model_name.endswith(
            "disable_thinking"
        ):
            return reason_text, answer
        return answer, reason_text


import requests
from requests.adapters import HTTPAdapter

# --- 全局变量区域 ---
_SESSION = requests.Session()
# 设置连接池大小为 4000，这是并发调用的关键
# pool_connections: 允许同时存在的连接池数量
# pool_maxsize: 允许连接池内缓存的最大连接数
adapter = HTTPAdapter(
    pool_connections=4000,
    pool_maxsize=4000,
)
_SESSION.mount("https://", adapter)


def response_siliconflow_LLM(messages, model_name, enable_thinking=False):
    """
    "Pro/deepseek-ai/DeepSeek-V3.2" RQM 4000; TPM 4e6
    :return {reason_text}{thinking_token_ending}{ans}
    """
    token_id = os.getenv("siliconflow_token_id")
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": 16384,
        "enable_thinking": enable_thinking,
        "thinking_budget": 16384,
        "min_p": 0.05,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.0,
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token_id}",
    }
    # 获取复用的 session
    session = _SESSION
    # 使用 session.post 发送请求（性能远高于 requests.request）
    response = session.post(url, json=payload, headers=headers, timeout=7200)
    response.raise_for_status()  # 如果状态码不是 200，抛出异常
    response_dict = response.json()  # 直接使用 .json() 更高效
    response_text = response_dict["choices"][0]["message"]["content"]
    if "reasoning_content" in response_dict["choices"][0]["message"]:
        reason_text = response_dict["choices"][0]["message"]["reasoning_content"]
        response_text = f"{reason_text}{thinking_token_ending}{response_text}"
    elif thinking_token_ending not in response_text:
        response_text = thinking_token_ending + response_text
    return response_text


# 用于启动vllm
vLLM_log_file_path = None


def launch_vllm_server(
    model_path="/data/pretrained/hf/Qwen/Qwen2.5-7B-Instruct",
    model_name="Qwen2.5-7B-Instruct",
):
    global vLLM_log_file_path
    """
    启动vLLM，先选择可用的port，然后通过命令行运行vLLM；
    循环等待，直到vLLM启动成功
    """
    import socket
    from utils.call_LLM import online_LLM_dict

    for port in range(8000, 8004):
        if (
            not os.path.exists(f"vllm_log-{port}.txt")
            and not os.path.exists(f"../benchmark/vllm_log-{port}.txt")
            and not os.path.exists(f"../auxiliary_benchmark/vllm_log-{port}.txt")
        ):
            continue
        online_LLM_dict[model_name] = {
            "API_key": "xxxxxxxxxxxxxxx",
            "base_URL": f"http://localhost:{port}/v1",
        }
        try:
            ans = response_from_online_LLM(
                [
                    {
                        "role": "user",
                        "content": "请你说这句话：“vllm启动成功。”",
                    }
                ],
                model_name=model_name,
            )
            print("✅ vLLM 已经启动", flush=True)
            return None
        except Exception as e:
            pass
    print("⚠️ vLLM 未启动，准备启动vLLM", flush=True)

    def find_free_port(start_port=8000, max_ports=100):
        """
        获取一个能用的port，从而实现vLLM同时部署多个LLM
        要求 端口可用 & 端口对应的log文件不存在，从而避免多个vLLM运行时冲突
        """
        global vLLM_log_file_path
        for port in range(start_port, start_port + max_ports):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    vLLM_log_file_path = f"vllm_log-{port}.txt"
                    # 另一个vLLM可能启动了，但是还没有占用port；仍然跳过这个port
                    if os.path.exists(vLLM_log_file_path):
                        continue
                    # 创建log文件，用于表明port已经被占用
                    open(vLLM_log_file_path, "w").close()
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"找不到可用端口（从 {start_port} 开始）")

    def modify_FSDP_config():
        """修改config.json中， FSDPQwen2ForCausalLM -> Qwen2ForCausalLM"""
        try:
            config_json_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_json_path):
                config = read_json(config_json_path)
                if config["architectures"][0].startswith("FSDP"):
                    config["architectures"][0] = config["architectures"][0][4:]
                    os.rename(
                        config_json_path,
                        os.path.join(model_path, "backup_config_FSDP.json"),
                    )
                    write_json(config, config_json_path)
        except:
            pass

    modify_FSDP_config()
    import torch

    tensor_parallel_size = torch.cuda.device_count()
    port = find_free_port()
    # 定义命令
    command = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        "0.75",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    print(" ".join(command), flush=True)

    # 打开日志文件
    log_file = open(vLLM_log_file_path, "w")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
        text=True,
    )
    online_LLM_dict[model_name] = {
        "API_key": "xxxxxxxxxxxxxxx",
        "base_URL": f"http://localhost:{port}/v1",
    }
    max_try_cnt = 240  # 限时20分钟内启动，否则视为失败
    while True:
        try:
            ans = response_from_online_LLM(
                [
                    {
                        "role": "user",
                        "content": "请你说这句话：“vllm启动成功。”",
                    }
                ],
                model_name=model_name,
            )
            print("✅ vLLM 启动成功")
            break
        except Exception as e:
            time.sleep(5)
            max_try_cnt -= 1
            if max_try_cnt <= 0:
                raise RuntimeError("vLLM启动失败")
            continue
    return process


def kill_vllm_server(vllm_process):
    """
    关闭vLLM，释放资源
    """
    if vllm_process is None:
        return
    vllm_process.terminate()
    try:
        vllm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        vllm_process.kill()
    time.sleep(5)
    # 检查是否已经终止
    if vllm_process.poll() is None:
        print("⚠️ vllm 终止失败")
    else:
        print("✅ vLLM 终止成功")
    # 通过重命名log文件，表达 端口已经释放
    os.rename(
        vLLM_log_file_path,
        f'vLLM_log_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt',
    )


class LLM_Caller_for_One_Thread:
    """
    record the log of one thread, parallel run
    """

    def __init__(
        self,
        introduction_log=None,
        dict_from_sys_prompt_to_context_id={},
        generation_kwargs={
            "max_tokens": 16384,
        },
    ) -> None:
        """
        introduction_log: 放在log中的第一句话
        dict_from_sys_prompt_to_context_id: prompt cache时，缓存的system prompt会对应一个id
        """
        self.LLM_log_list = []
        if introduction_log:
            self.LLM_log_list.append(introduction_log)
        # system_prompt may be repeated and cached
        self.dict_from_sys_prompt_to_context_id = dict_from_sys_prompt_to_context_id
        self.generation_kwargs = generation_kwargs
        self.try_seed = 42 if "seed" not in generation_kwargs else None

    def query_model_with_reasoning(
        self,
        model_str,
        prompt=None,
        system_prompt=None,
        messages=None,
        role="Patient",
        ensure_label="answer",
        try_cnt=6,
        timeout=2.0,
        generation_kwargs={},
    ):
        """
        call LLM and record log
        ensure_label: make sure the format is <ensure_label>...</ensure_label>
        return  reasoning_content, answer
        """
        # try_cnt = 30
        # if "gpt" in model_str:
        #     try_cnt = 10
        if not generation_kwargs:
            generation_kwargs = self.generation_kwargs.copy()
        if messages:
            system_prompt = messages[0]["content"]
            prompt = messages[-1]["content"]
        else:
            messages = []
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}]
            messages += [{"role": "user", "content": prompt}]
        Exception_list = []
        for _ in range(try_cnt):
            try:
                if self.try_seed:
                    # 每次生成，使用不同的seed
                    generation_kwargs["seed"] = self.try_seed + _
                answer, reasoning_content = response_from_online_LLM(
                    messages,
                    model_str,
                    split_reason_content=True,
                    generation_kwargs=generation_kwargs,
                )
                if answer is None or answer == "":
                    if len(reasoning_content) > 0:
                        answer = reasoning_content
                        reasoning_content = ""
                markdown_log_str = (
                    "---\n\n"
                    + f"""**role**: {role}\n"""
                    + f"""**time** {datetime.datetime.now().strftime("%Y%m%d-%H%M")}\n"""
                    + f"""**model_name** {model_str}\n"""
                    + f"""**messages_length** {len(messages)}\n"""
                    + f"""**system_prompt** \n```\n{system_prompt}\n```\n"""
                    + f"""**user_prompt** \n```\n{prompt}\n```\n"""
                    + f"""**reasoning_content** \n```\n{reasoning_content}\n```\n"""
                    + f"""**answer** \n```\n{answer}\n```\n"""
                    + "\n\n\n"
                )
                self.LLM_log_list.append(markdown_log_str)
                if os.environ.get("DEBUG", "0") == "1":
                    print(markdown_log_str, flush=True)

                # DeepSeek 偶尔不输出</answer>标签，导致提取失败
                if (
                    ensure_label == "answer"
                    and "</answer>" not in answer
                    and answer.count("<answer>") == 1
                    and len(answer.split("<answer>")[-1]) < 500
                ):
                    answer = answer + "</answer>"

                # Baichuan-M3-235B 经常不输出<answer>标签，导致提取失败
                if (
                    "baichuan" in model_str.lower()
                    and ensure_label == "answer"
                    and "</answer>" not in answer
                    and "<answer>" not in answer
                    and len(answer) < 600
                ):
                    answer = "<answer>" + answer + "</answer>"

                if ensure_label is not None:
                    extracted_ans = extract_label(answer, ensure_label)
                    if extracted_ans is None or len(extracted_ans) == 0:
                        # not good format, re-generate
                        raise ValueError(
                            "格式不正确: extracted_ans(not None)=",
                            extracted_ans,
                            "answer=",
                            answer,
                        )
                reasoning_content = (
                    reasoning_content.replace("<think>", "")
                    .replace("</think>", "")
                    .strip()
                )
                # # 判断是否有重复某些句子的死循环
                # if len(reasoning_content) > 2000:
                #     p = reasoning_content[-10:]
                #     if reasoning_content.count(p) > 30:
                #         continue
                # if len(answer) > 2000:
                #     p = answer[-10:]
                #     if answer.count(p) > 30:
                #         continue
                return reasoning_content, answer.strip()

            except Exception as e:
                if os.environ.get("DEBUG", "0") == "1":
                    print(f"Error: {e}\n{system_prompt}\n{prompt}", flush=True)
                time.sleep(random.uniform(0, timeout))
                Exception_list.append(e)
                continue
        # exception去重
        Exception_list = list(set(Exception_list))
        raise Exception(
            f"Max retry_cnt: {model_str}\n\n{str(Exception_list)}\n\n{system_prompt}\n{prompt[:]}"
        )

    def query_model(
        self,
        model_str,
        prompt=None,
        system_prompt=None,
        messages=None,
        role="Patient",
        ensure_label="answer",
        try_cnt=6,
        timeout=2.0,
        generation_kwargs={},
    ):
        """
        return  answer
        """
        reasoning_content, answer = self.query_model_with_reasoning(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            role=role,
            ensure_label=ensure_label,
            try_cnt=try_cnt,
            timeout=timeout,
            generation_kwargs=generation_kwargs,
        )
        return answer

    def query_model_and_extract_label(
        self,
        model_str,
        prompt=None,
        system_prompt=None,
        messages=None,
        role="Patient agent",
        ensure_label="answer",
        try_cnt=6,
        timeout=2.0,
        generation_kwargs={},
    ):
        assert ensure_label is not None
        temp = self.query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            role=role,
            ensure_label=ensure_label,
            try_cnt=try_cnt,
            timeout=timeout,
            generation_kwargs=generation_kwargs,
        )
        return extract_label(temp, ensure_label)
