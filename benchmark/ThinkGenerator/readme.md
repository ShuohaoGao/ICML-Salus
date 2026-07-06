## 构造数据的基本流程 `main.py`

### 运行方法

其他参数定义在 `config.py` 中。

#### 示例命令（使用 GLM 构造数据；前 10 个病历；240 个线程）：
```sh
nohup python main.py \
    --dataset_path="../OSCE-extract/爱医医/iiy-refined_train.json" \
    --action_llm="GLM-Z1-Flash" \
    --thinker_llm="GLM-Z1-Flash" \
    --moderator_llm="GLM-Z1-Flash" \
    --inquiry_num=30 \
    --physical_exam_num=40 \
    --auxiliary_exam_num=10 \
    --num_scenarios=10 \
    --parallel_thread_num=240 \
    --repeat_num=5 \
    --Chinese \
    > GLM-Z1-Flash-爱爱医.md &
```
其中， `action_llm, thinker_llm, moderator_llm`的角色分别如下：
- `action_llm`: 生成医生的下一步动作
- `thinker_llm`: 生成思维链
- `moderator_llm`: 其他任务，例如判断下一步动作是否在病历中有依据、基于生成的对话进行诊断等

其设置可以参考`utils/call_LLM.py`中的`online_LLM_dict`，一般可以使用以下几个：
- deepseek-r1
- deepseek-v3-0324
- GLM-Z1-Flash (推理模型，免费)
- GLM-4-Flash (非推理模型，免费)
- qwen3-32b

也可以直接使用本地模型，先启动vllm进程，然后在`online_LLM_dict`中更新模型的名字。
```
先启动vllm: CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_MODELSCOPE=true vllm serve /data/pretrained/hf/Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --tensor-parallel-size 4

然后向utils/call_LLM.py 中 online_LLM_dict添加：

"Qwen2.5-7B-Instruct": {
    "API_key": "xxxxxxxxxxxxxxx",
    "base_URL": "http://localhost:8000/v1",
}, 

然后就可以调用本地LLM了：
    --action_llm="Qwen2.5-7B-Instruct"
```


#### 输出文件说明：
- 主输出日志：`GLM-Z1-Flash-爱爱医.md`
- 详细训练数据日志：
  - JSON 格式：`detailed_log/时间-病历文件--doctor=GLM-Z1-Flash.json`
  - Markdown 日志：`detailed_log/时间-病历文件--doctor=GLM-Z1-Flash.md`

- [x] JSON 文件存储的是 messages 格式的训练数据  
- [ ] MD 文件记录了 LLM 调用的完整日志，一般用不到

---

### 函数入口

- **主函数**：`run_one_scenario`
- **输入对象**：`scenario: Scenario_OSCE`
- **并行处理机制**：
  - 每条多轮对话对应一个独立线程
  - 每个病历会重复构造 5 条对话以增强多样性

---

### 数据构造流程详解

#### 1. 构建对话

- **调用函数**：`generate_conversation`
- **输出示例**：
```py
messages = [
    {
        "role": "user",
        "content": "[主诉内容]",
        "stage": "Inquiry"
    },
    {
        "role": "assistant",
        "content": "[医生问诊]",
        "stage": "Inquiry"
    },
    ...
    {
        "role": "assistant",
        "content": "[最终诊断]",
        "stage": "Diagnosis"
    }
]
```

> 📌 `stage` 字段包含以下四种状态：
> - `Inquiry`：问诊阶段
> - `Physical Exam`：体检阶段
> - `Auxiliary Exam`：辅检阶段
> - `Diagnosis`：诊断阶段

- **具体逻辑**：
  - 分为 3 个阶段，每个阶段使用病历中对应的部分信息进行生成

---

#### 2. 构建思维链

- **调用函数**：`generate_thinking`
- **输入参数**：`messages`（即上一步 `generate_conversation` 的输出）
- **输出示例**：
```py
messages = [
    {
        "role": "user",
        "content": "[主诉内容]",
        "stage": "Inquiry"
    },
    {
        "role": "assistant",
        "reasoning_content": "构造的思维链",  # 新生成的部分
        "content": "[医生问诊]",
        "stage": "Inquiry"
    },
    ...
    {
        "role": "assistant",
        "reasoning_content": "构造的思维链",  # 新生成的部分
        "content": "[最终诊断]",
        "stage": "Diagnosis"
    }
]
```

---

#### 3. 对话质量检验（诊断一致性验证）

- **调用函数**：`make_diagnosis_and_verify`
- **逻辑说明**：
  - 使用构造出的对话历史（去掉最后一条“诊断”语句）
  - 再次让 LLM 进行推理并输出诊断结果
  - 判断是否与原始诊断一致，用于评估生成质量

---

## 工具函数说明

### 1. LLM 调用工具类：`LLM_caller: LLM_Tools_for_One_Thread`

- **特点**：
  - 每个线程使用独立的 `LLM_caller` 实例
  - 自动记录每次调用的日志

- **调用接口，单轮对话传入user/system prompt**：
```py
answer = LLM_caller.query_model(
    model_str=args.doctor_llm,
    prompt="...",           # user prompt
    system_prompt="...",
    role="Doctor",
    ensure_label="answer",  # 强制要求输出格式如 `<answer>xxx</answer>`，否则重试
)
```

#### 多轮对话调用，传入messages：
```py
answer = LLM_caller.query_model(
    model_str=args.doctor_llm,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        ...
    ],
    role="Doctor",
    ensure_label="answer"
)
```

---

### 2. Prompt 模块：`prompt_Chinese.py` / `prompt_English.py`

- **加载方式**：
```py
from prompt_English import *     # 默认加载英文 prompt

if args.Chinese:
    from prompt_Chinese import * # 若启用中文，则切换为中文 prompt
```

#### 包含的 Prompt 类型：
- **思维链生成**
  - `SYSTEM_prompt_generate_thinking`
  - `USER_prompt_generate_thinking`
- **医生行为预测（分阶段）**
  - user prompt 统一：`USER_prompt_action`
  - 不同阶段system prompt：
    - 问诊阶段：`SYSTEM_prompt_action_inquiry`
    - 体检阶段：`SYSTEM_prompt_action_physical`
    - 辅检阶段：`SYSTEM_prompt_action_auxiliary`


## 其他文件说明
- `data_loader.py`: `Scenario_OSCE`对象存储一份病历
- `config.py`: 命令行参数；`args`是全局变量