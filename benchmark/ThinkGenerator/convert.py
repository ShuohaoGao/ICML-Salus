import sys

sys.path.append("../")
from utils.tools import extract_label, batch_process_parallel, analyze_distribution
from utils.io_func import write_json, read_json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
)


def compute_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def build_chatml(messages):
    chatml_text = ""
    chatml_text += f"<|im_start|>system\n{messages['system']}<|im_end|>\n"
    chatml_text += f"<|im_start|>user\n{messages['messages'][0]['content']}<|im_end|>\n"
    chatml_text += (
        f"<|im_start|>assistant\n{messages['messages'][1]['content']}<|im_end|>\n"
    )
    return chatml_text


def get_token(messages):
    return compute_tokens(build_chatml(messages))


def stage_cnt(messages):
    cnt = {}
    for m in messages:
        if m["role"] == "user":
            continue
        stage = m["stage"]
        cnt[stage] = cnt.get(stage, 0) + 1
    return cnt


action_inquiry_physical_system_prompt = (
    "你负责根据提供的病人信息，决定医生下一步要做的问诊或体格检查。"
)
action_inquiry_physical_user_prompt = (
    "<病历>\n{}\n</病历>\n\n<上一步动作>\n{}\n</上一步动作>"
)

long_diagnosis_system_prompt = (
    "你是一名专业医生，负责根据提供的病人信息，进行鉴别诊断。"
)
diagnosis_user_prompt = "<病历>\n{}\n</病历>"

action_auxiliary_item_system_prompt = "你是一名专业医生，负责根据提供的病人信息和鉴别诊断，决定医生下一步要进行哪些辅助检查"
action_exam_or_diagnosis = "你是一名专业医生，负责根据提供的病人信息和鉴别诊断，决定医生下一步是继续进行辅助检查，还是直接给出最终诊断。"
action_user_prompt = "<病历>\n{}\n</病历>\n\n<鉴别诊断>\n{}\n</鉴别诊断>"

verify_system_prompt = "你负责验证指定的诊断是否与病人的信息矛盾。若无矛盾，输出yes"
verify_user_prompt = "<病历>\n{}\n</病历>\n\n<诊断>\n{}\n</诊断>"


def get_train_format_data(generated_messages: list):

    def format_message(
        system_prompt, user_prompt, reason, answer, reason_label="reason"
    ):
        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"<{reason_label}>{reason}</{reason_label}>{answer}",
                },
            ],
            "system": system_prompt,
        }

    data_list = []
    for idx, message in enumerate(generated_messages):
        if message["role"] != "assistant":
            continue
        if "action_reasoning_inquiry+physical" in message:
            # 问诊体检动作
            action_reason = message["action_reasoning_inquiry+physical"]
            last_action = message["last_action"]
            data_list.append(
                format_message(
                    system_prompt=action_inquiry_physical_system_prompt,
                    user_prompt=action_inquiry_physical_user_prompt.format(
                        message["summary"], last_action
                    ),
                    reason=action_reason,
                    answer=(
                        "我现在终止体格检查，开始进入辅助检查阶段。"
                        if "终止体格检查，开始进入辅助检查阶段" in action_reason
                        else message["content"]
                    ),
                )
            )
        if "long_diagnosis" in message:
            long_diagnosis_reason, long_diagnosis_answer = message["long_diagnosis"]
            # 详细诊断分析
            data_list.append(
                format_message(
                    system_prompt=long_diagnosis_system_prompt,
                    user_prompt=diagnosis_user_prompt.format(message["summary"]),
                    reason=long_diagnosis_reason,
                    answer=long_diagnosis_answer,
                )
            )

            # 继续检查还是诊断
            if "decision_exam_or_diagnosis" in message:
                decision_reason, decision_answer = message["decision_exam_or_diagnosis"]
                data_list.append(
                    format_message(
                        system_prompt=action_exam_or_diagnosis,
                        user_prompt=action_user_prompt.format(
                            message["summary"],
                            message["verified_diagnosis"],
                        ),
                        reason=decision_reason,
                        answer=decision_answer,
                    )
                )

            # 辅助检查推荐
            if "auxiliary_exam_reasoning" in message:
                auxiliary_exam_reason, auxiliary_exam_answer = message[
                    "auxiliary_exam_reasoning"
                ]
                data_list.append(
                    format_message(
                        system_prompt=action_auxiliary_item_system_prompt,
                        user_prompt=action_user_prompt.format(
                            message["summary"],
                            message["verified_diagnosis"],
                        ),
                        reason=auxiliary_exam_reason,
                        answer=auxiliary_exam_answer,
                    )
                )

            # 地毯式验证
            if "verify_list" in message:
                for d in message["verify_list"]:
                    data_list.append(
                        format_message(
                            system_prompt=verify_system_prompt,
                            user_prompt=verify_user_prompt.format(
                                message["summary"], d["diagnosis"]
                            ),
                            reason=d["verify_text"],
                            answer="yes" if d["verify_result"] else "no",
                        )
                    )

    return data_list


input_path_list = [
    "./detailed_log/20251220-1703-train_set--doctor=deepseek-v3-2-251201-enable_thinking.json",
]
dataset_name = "train_data_1221_Deepseek-V3-2_1repeats_All"
raw_data = []
for input_path in input_path_list:
    raw_data += read_json(input_path)["results_list"]
print("病历个数：", len(raw_data))
gathered_data_list = batch_process_parallel(
    func=get_train_format_data,
    args_list=[
        [x["generated_messages"]]
        for x in raw_data
        if x["generated_messages"] and x["success"]
    ],
    num_processes=200,
    use_tqdm=True,
)
all_data_list = []
for x in gathered_data_list:
    all_data_list.extend(x)
print("训练数据条数：", len(all_data_list))

token_list = batch_process_parallel(
    func=get_token,
    args_list=[[m] for m in all_data_list],
    num_processes=200,
    use_tqdm=True,
)
print("-" * 30, "Tokens 分布")
analyze_distribution(token_list)


stage_cnt_list = batch_process_parallel(
    func=stage_cnt,
    args_list=[[x["generated_messages"]] for x in raw_data if x["success"]],
    batch_size=100,
    num_processes=100,
    use_tqdm=False,
)
print("-" * 30, "问诊次数分布")
inquiry_cnt_list = [x.get("Inquiry", 0) for x in stage_cnt_list]
analyze_distribution(
    inquiry_cnt_list, percentiles=[50, 75, 95, 99], thresholds=[5, 8, 10]
)
print("-" * 30, "体检次数分布")
physical_cnt_list = [x.get("Physical Exam", 0) for x in stage_cnt_list]
analyze_distribution(
    physical_cnt_list, percentiles=[50, 75, 95, 99], thresholds=[5, 10, 15]
)
print("-" * 30, "辅检次数分布")
auxiliary_cnt_list = [x.get("Auxiliary Exam", 0) for x in stage_cnt_list]
analyze_distribution(
    auxiliary_cnt_list, percentiles=[50, 75, 95, 99], thresholds=[3, 4, 5, 6, 10]
)


print("-" * 30, "指令类型分布")
instruct_type_cnt = {}
for m in all_data_list:
    sys_p = m["system"]
    instruct_type_cnt[sys_p] = instruct_type_cnt.get(sys_p, 0) + 1
for k, v in instruct_type_cnt.items():
    print(str(k).ljust(100), v)

write_json(
    all_data_list,
    f"/data/InteractiveMedLLM/train_data/OpenAI_format/{dataset_name}.json",
)


# 创建datasets格式
def trl_format(data: dict):
    return {
        "messages": [{"role": "system", "content": data["system"]}] + data["messages"]
    }


processed_data_list = batch_process_parallel(
    func=trl_format,
    args_list=[[x] for x in all_data_list],
    batch_size=100,
    num_processes=100,
    use_tqdm=False,
)
print(f"processed_data_list 类型: {type(processed_data_list)}")
print(f"第一个元素的类型: {type(processed_data_list[0])}")
print(f"第一个元素的内容: {processed_data_list[0]}")

from datasets import Dataset, DatasetDict

full_dataset = Dataset.from_list(processed_data_list)
# test_size=0.01 表示将1%的数据划为测试集（我们将用作验证集）
split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
print("数据集划分完成:")
print(split_dataset)
output_path = f"/data/InteractiveMedLLM/train_data/local_datasets/{dataset_name}"
split_dataset.save_to_disk(output_path)

print(f"\n数据集已成功保存到: {output_path}")
