import sys

sys.path.append("../")
from utils.call_LLM import (
    LLM_Caller_for_One_Thread,
    get_context_id,
    cached_LLM_id_list,
)
from utils.tools import extract_label
from utils.io_func import write_json, read_json
import re, time
from datetime import datetime
from utils.data_loader import Scenario_OSCE_Loader, Scenario_OSCE
from pathlib import Path
import random
import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from config_generator import args
from prompt_English import *

if args.Chinese:
    from prompt_Chinese import *

global_dict_from_sys_prompt_to_context_id = {}


def transform_messages_to_conversation_str(messages: list) -> str:
    """
    把messages拼接为对话字符串
    messages:
    [
        {
            "role": "user",
            "content": "...",
            "stage": "Inquiry",
        },
        {
            "role": "assistant",
            "content": "...",
            "stage": "Inquiry",
        },
        ...
    ]
    """
    conversation_str = ""
    for message in messages:
        if message["role"] == "assistant":
            conversation_str += f"Doctor: {message['content']}\n"
            continue
        # Patient, role=user
        if message["stage"] == "Inquiry":
            conversation_str += f"Patient: {message['content']}\n\n"
        elif message["stage"] == "Physical Exam":
            conversation_str += f"Results: {message['content']}\n\n"
        elif message["stage"] == "Auxiliary Exam":
            conversation_str += f"Results: {message['content']}\n\n"
    return conversation_str


def is_valid_messages(messages):
    # 检查合理性，user assistant交替出现
    last_role = "assistant"
    for item in messages:
        role = item["role"]
        if role == last_role:
            return False
        last_role = role
        if "content" not in item:
            return False
    return True


def judge_correct_diagnosis_level(
    correct_diagnosis: str,
    diagnosis_to_be_judged: str,
    LLM_caller: LLM_Caller_for_One_Thread,
    other_info=None,
    model_name: str = "GLZ-Z1-Flash",
    level_list: list = ["完全正确", "临床可接受", "部分正确", "不正确"],
) -> tuple[bool, str]:
    if not diagnosis_to_be_judged or not correct_diagnosis:
        return False, level_list[-1]
    try:
        if not other_info:
            other_info = "空"
        prefix = (
            f"最终诊断 (绝对正确的标准答案)：\n{correct_diagnosis}\n\n"
            + f"医生诊断 (有待评估):\n{diagnosis_to_be_judged}\n\n"
            # + f"\n\n病历信息:\n{other_info}\n\n"
        )
        for _ in range(3):
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=USER_PROMPT_judge_diagnosis_correct_level.format(prefix),
                system_prompt=SYSTEM_PROMPT_judge_diagnosis_correct_level,
                role="Judge Diagnosis Level",
                ensure_label="answer",
            )
            if ans in level_list:
                correctness = ans in ["完全正确", "临床可接受"]
                return correctness, ans
    except Exception as e:
        print("Error in judge_correct_diagnosis_level:", e, flush=True)
    return False, level_list[-1]


def compare_results(
    diagnosis_to_be_verified, correct_diagnosis, LLM_caller: LLM_Caller_for_One_Thread
) -> bool:
    """evaluate whether the diagnosis is correct"""
    correctness, level = judge_correct_diagnosis_level(
        correct_diagnosis=correct_diagnosis,
        diagnosis_to_be_judged=diagnosis_to_be_verified,
        LLM_caller=LLM_caller,
        model_name=args.cheap_llm,
    )
    return correctness


def generate_conversation_inquiry_and_physical_exam(
    scenario: Scenario_OSCE, LLM_caller: LLM_Caller_for_One_Thread
):
    inquiry_messages = []
    # 1. 问诊，一次性生成完整对话
    for _ in range(3):
        conversation_inquiry = LLM_caller.query_model_and_extract_label(
            model_str=args.expensive_llm,  # 这里使用了较强的LLM避免幻觉
            prompt=prompt_action_inquiry_all.format(
                scenario.chief_complaint_info().replace("\n", " "),
                scenario.patient_information(),
            ),
            role="Action: Inquiry dialog",
            ensure_label="answer",
        )
        if conversation_inquiry:
            # 检查幻觉
            hallucination = LLM_caller.query_model_and_extract_label(
                model_str=args.cheap_llm,
                prompt=prompt_check_hallucination.format(conversation_inquiry),
                role="Action: check dialog hallucination",
                ensure_label="answer",
            )
            if hallucination and hallucination.lower().startswith("no"):
                for line in conversation_inquiry.split("\n"):
                    line = line.lower()
                    if line.startswith("patient:"):
                        line = line.replace("patient:", "").strip()
                        inquiry_messages.append(
                            {
                                "role": "user",
                                "content": line,
                                "stage": "Inquiry",
                            }
                        )
                    elif line.startswith("doctor:"):
                        line = line.replace("doctor:", "").strip()
                        inquiry_messages.append(
                            {
                                "role": "assistant",
                                "content": line,
                                "stage": "Inquiry",
                            }
                        )
                # 检查格式
                if is_valid_messages(inquiry_messages) and len(inquiry_messages) > 8:
                    break
                else:
                    inquiry_messages = []
    if len(inquiry_messages) <= 1:
        raise ValueError("问诊对话构造失败，可能是有幻觉")
    if inquiry_messages[-1]["role"] == "assistant":
        inquiry_messages.pop()

    # 2. 体格检查：一次性生成多组
    conversation_inquiry = transform_messages_to_conversation_str(inquiry_messages)
    physical_exam_messages = []
    for _ in range(3):
        temp_output = LLM_caller.query_model(
            model_str=args.cheap_llm,
            prompt=USER_prompt_action_physical_exam_all_groups.format(
                conversation_inquiry, scenario.physical_exam_information()
            ),
            system_prompt=SYSTEM_prompt_action_physical_exam_all_groups,
            role="Action: Physical exam dialog",
            ensure_label="1",
        )
        cnt_invalid_name = 0
        if "<25>" not in temp_output:  # 不要太多轮数
            for idx in range(1, 25):
                group_conversation = extract_label(temp_output, str(idx))
                if not group_conversation:
                    continue
                # 放入messages
                doctor_line = (
                    group_conversation.split("Patient:")[0].split("Doctor:")[-1].strip()
                )
                patient_line = group_conversation.split("Patient:")[-1].strip()
                if not doctor_line or not patient_line:
                    continue
                item_name = doctor_line.split("请求一组体格检查：")[-1].strip()
                if item_name not in [
                    "一般状况与生命体征（意识状态、生命体征、整体外观、身高体重）",
                    "皮肤与黏膜系统（皮肤颜色、皮疹病变、皮下结节、水肿、皮温湿度、黏膜）",
                    "浅表淋巴结检查（头颈、腋窝、腹股沟淋巴结）",
                    "头颅与面部（头颅检查、面部检查）",
                    "眼部检查（视功能、眼外部、眼球）",
                    "耳鼻咽喉检查（耳部、鼻部、口咽部）",
                    "颈部与甲周腺（颈部外形活动度、甲状腺、颈部血管、气管）",
                    "胸部视触叩听（视诊、听诊、触诊、叩诊）",
                    "心脏视触叩听（视诊、叩诊、听诊、触诊）",
                    "腹部检查（视诊、触诊、叩诊、听诊、肝脾、肾区膀胱）",
                    "脊柱检查（外形、触诊、活动度）",
                    "关节检查（红肿、压痛、活动度、畸形）",
                    "四肢检查（上肢、下肢）",
                    "脑神经功能（12对脑神经）",
                    "运动系统检查（肌力、肌张力、肌肉萎缩、不自主运动）",
                    "感觉系统检查（浅感觉、深感觉、复合感觉）",
                    "反射功能检查（深反射、浅反射、病理反射）",
                    "协调和平衡功能（平衡协调、步态分析）",
                    "男性生殖器检查（外生殖器、腹股沟、前列腺）",
                    "女性生殖器检查（外阴尿道、阴道宫颈、子宫附件）",
                    "肛门与直肠检查（肛周、直肠指诊）",
                    "外周血管检查（动脉、静脉、末梢循环）",
                    "妇产科检查（妊娠期、乳房）",
                    "儿科检查（生长发育、新生儿专项）",
                    "老年人检查（功能评估、风险筛查）",
                ]:
                    cnt_invalid_name += 1
                    continue
                physical_exam_messages.append(
                    {
                        "role": "assistant",
                        "content": doctor_line,
                        "stage": "Physical Exam",
                    }
                )
                physical_exam_messages.append(
                    {
                        "role": "user",
                        "content": patient_line,
                        "stage": "Physical Exam",
                    }
                )
            # 检查格式
            if (
                len(physical_exam_messages) >= 2
                and len(physical_exam_messages) % 2 == 0
                and cnt_invalid_name <= 1
            ):
                break
            else:
                physical_exam_messages = []

    if len(physical_exam_messages) <= 1:
        raise ValueError("体检对话构造失败")
    if physical_exam_messages[-1]["role"] == "assistant":
        physical_exam_messages.pop()

    generated_messages = inquiry_messages + physical_exam_messages
    if not is_valid_messages(generated_messages):
        raise ValueError("生成体检对话后 messages 格式不对")

    return generated_messages


def generate_thinking_inquiry_and_physical_exam(
    generated_messages: list, LLM_caller: LLM_Caller_for_One_Thread
):
    """
    2种数据：
    1. 总结
    2. 思考下一步
    """
    conversation_summary = "问诊结果：\n"
    last_action = "尚未进行问诊"
    for idx, message in enumerate(generated_messages):
        if message["role"] == "user":  # 病人的回复，进行总结
            if (
                idx >= 2
                and generated_messages[idx - 2]["stage"] == "Inquiry"
                and message["stage"] == "Physical Exam"
            ):  # 问诊 → 体检
                conversation_summary += "\n体格检查结果：\n"
            # 只总结最新一轮的QA
            recent_question_answer = transform_messages_to_conversation_str(
                generated_messages[max(0, idx - 1) : idx + 1]
            )
            try:
                temp_summary = LLM_caller.query_model_and_extract_label(
                    model_str=args.cheap_llm,
                    prompt=prompt_summary.format(recent_question_answer),
                    role="summary",
                    ensure_label="answer",
                )
            except:
                temp_summary = recent_question_answer
            conversation_summary += temp_summary
            message["summary"] = conversation_summary
            if random.uniform(0, 1) > 0.5:  # 有一定概率不总结，让LLM见到多种输出格式
                message["summary"] = transform_messages_to_conversation_str(
                    generated_messages[: idx + 1]
                )
            continue
        # assistant，即医生部分
        if message["stage"] in ["Inquiry", "Physical Exam"]:
            message["summary"] = conversation_summary  # 总结 = 上次病人回答后的总结
            if random.uniform(0, 1) > 0.5:  # 有一定概率不总结，让LLM见到多种输出格式
                message["summary"] = transform_messages_to_conversation_str(
                    generated_messages[:idx]
                )
            next_action = message["content"]
            action_reasoning = LLM_caller.query_model_and_extract_label(
                model_str=args.cheap_llm,
                prompt=prompt_next_action.format(
                    message["summary"], last_action, next_action
                ),
                role="next action reasoning",
                ensure_label="answer",
            )
            message["action_reasoning_inquiry+physical"] = action_reasoning
            message["last_action"] = last_action
            last_action = next_action
    return generated_messages


def generate_auxiliary_exam(
    scenario: Scenario_OSCE,
    generated_messages: list,
    summary_inquiry_physical: str,
    LLM_caller: LLM_Caller_for_One_Thread,
    last_physical_action: str = "一般状况与生命体征（意识状态、生命体征、整体外观、身高体重）",
    step_by_step_ratio=0.0,
):
    # 体检到辅助检查：先简单诊断，然后判断需要转到辅检阶段；然后详细诊断，然后选择辅检项目
    first_auxiliary_message = {
        "role": "assistant",
        "content": "",
        "stage": "Auxiliary Exam",
    }
    conversation_summary = summary_inquiry_physical
    if True:  # 体检 -> 辅检
        first_auxiliary_message["summary"] = conversation_summary

        next_action = "终止体格检查，开始进入辅助检查阶段。"
        action_reasoning = LLM_caller.query_model_and_extract_label(
            model_str=args.cheap_llm,
            prompt=prompt_next_action.format(
                first_auxiliary_message["summary"],
                last_physical_action,
                next_action,
            ),
            role="next action reasoning",
            ensure_label="answer",
        )
        first_auxiliary_message["action_reasoning_inquiry+physical"] = action_reasoning
        first_auxiliary_message["last_action"] = last_physical_action

    num_inquiry_physical = len(generated_messages)
    conversation_summary += "\n辅助检查结果：\n"
    auxiliary_dialog_str = f"{conversation_summary}"
    for auxiliary_cnt in range(args.auxiliary_exam_num):
        is_first_auxiliary = (
            auxiliary_cnt * 2 + num_inquiry_physical == num_inquiry_physical
        )
        message = {
            "role": "assistant",
            "content": "",
            "stage": "Auxiliary Exam",
        }
        if is_first_auxiliary:
            message = first_auxiliary_message
        else:
            # 总结新一轮辅助检查的QA
            conversation_history = transform_messages_to_conversation_str(
                generated_messages[-2:]
            )
            try:
                temp_summary = LLM_caller.query_model_and_extract_label(
                    model_str=args.cheap_llm,
                    prompt=prompt_summary.format(conversation_history),
                    role="summary",
                    ensure_label="answer",
                )
            except:
                temp_summary = generated_messages[-1]["content"]
            conversation_summary += temp_summary
            message["summary"] = conversation_summary
            if random.uniform(0, 1) > 0.5:
                message["summary"] = auxiliary_dialog_str

        # 鉴别诊断
        for _ in range(3):
            long_diagnosis_reasoning_content, long_diagnosis_answer = (
                LLM_caller.query_model_with_reasoning(
                    model_str=args.expensive_llm,
                    prompt=prompt_long_diagnosis.format(message["summary"]),
                    role="long differential diagnosis",
                    ensure_label=None,
                )
            )
            if (
                long_diagnosis_answer.strip().startswith("- ")
                and long_diagnosis_reasoning_content
            ):
                break
        if not (
            long_diagnosis_answer.strip().startswith("- ")
            and long_diagnosis_reasoning_content
        ):
            raise ValueError("generate_auxiliary_exam 生成鉴别诊断失败")
        message["long_diagnosis"] = [
            long_diagnosis_reasoning_content,
            long_diagnosis_answer,
        ]
        diagnosis_list = [
            d.strip() for d in long_diagnosis_answer.split("\n") if d.strip()
        ]
        if len(diagnosis_list) == 0:
            raise ValueError("generate_auxiliary_exam 提取topk诊断失败")

        # 地毯式验证
        if random.uniform(0, 1) < step_by_step_ratio:
            verify_list = []
            verified_diagnosis_list = []
            for diagnosis in diagnosis_list:
                for _ in range(3):
                    temp_output = LLM_caller.query_model(
                        model_str=args.cheap_llm,
                        prompt=prompt_verify_diagnosis.format(
                            message["summary"], diagnosis
                        ),
                        role="verify diagnosis",
                        ensure_label="answer",
                    )
                    verify_text = extract_label(temp_output, "answer")
                    verify_result = extract_label(temp_output, "result")
                    if verify_text and verify_result:
                        verify_result = verify_result.lower().startswith("yes")
                        verify_list.append(
                            {
                                "diagnosis": diagnosis,
                                "verify_text": verify_text,
                                "verify_result": verify_result,
                            }
                        )
                        if verify_result:
                            verified_diagnosis_list.append(diagnosis)
                        elif diagnosis == scenario.core_diagnosis_info():
                            raise ValueError(
                                "generate_auxiliary_exam 地毯式验证失败，正确诊断被否定"
                            )
                        break
            message["verify_list"] = verify_list
            if len(verified_diagnosis_list) == 0:
                verified_diagnosis_list.append("- " + scenario.core_diagnosis_info())
            verified_diagnosis_str = "\n".join(verified_diagnosis_list)
        else:
            verified_diagnosis_str = "\n".join(diagnosis_list)

        message["verified_diagnosis"] = verified_diagnosis_str

        # 先判断是否继续检查；如果选择停止检查、给出诊断，那么必须诊断正确才保留构造数据
        valid_answer = False
        for _try_cnt in range(2):
            decision_reason, decision_answer = LLM_caller.query_model_with_reasoning(
                model_str=args.expensive_llm,
                prompt=prompt_exam_or_diagnosis.format(
                    message["summary"], verified_diagnosis_str
                ),
                role="decide exam or diagnosis",
                ensure_label=None,
            )
            if not decision_answer == "继续辅助检查" and not (
                decision_answer.startswith("您的诊断结果为：")
                and "您的完整诊断如下：" in decision_answer
            ):  # 格式不对
                continue
            if decision_answer.startswith("您的诊断结果为：") and compare_results(
                diagnosis_to_be_verified=decision_answer,
                correct_diagnosis=scenario.diagnosis_information(),
                LLM_caller=LLM_caller,
            ):  # 选择诊断，并且诊断正确
                valid_answer = True
                break
            if decision_answer == "继续辅助检查":
                valid_answer = True
                break
        if not valid_answer:
            print("跳过判断是否继续检查")
            continue_exam = True
        else:
            message["decision_exam_or_diagnosis"] = [decision_reason, decision_answer]
            continue_exam = decision_answer == "继续辅助检查"
        if not continue_exam:  # 诊断
            message["content"] = decision_answer
            message["stage"] = "Diagnosis"
            generated_messages.append(message)
            return generated_messages
        else:  # 进行辅助检查
            for _ in range(3):
                auxiliary_exam_reasoning_content, auxiliary_exam_answer = (
                    LLM_caller.query_model_with_reasoning(
                        model_str=args.expensive_llm,
                        prompt=prompt_recommend_auxiliary_exams.format(
                            message["summary"], verified_diagnosis_str
                        ),
                        role="recommend auxiliary exams",
                        ensure_label=None,
                    )
                )
                if (
                    auxiliary_exam_answer.startswith("请求进行以下辅助检查：")
                    and auxiliary_exam_reasoning_content
                ):
                    break
            if not (
                auxiliary_exam_answer.startswith("请求进行以下辅助检查：")
                and auxiliary_exam_reasoning_content
            ):
                raise ValueError("生成下一步辅助检查 失败")
            message["content"] = auxiliary_exam_answer
            message["auxiliary_exam_reasoning"] = [
                auxiliary_exam_reasoning_content,
                auxiliary_exam_answer,
            ]
            generated_messages.append(message)
            # 添加病人回复：检查结果
            results_auxiliary_exam = LLM_caller.query_model_and_extract_label(
                model_str=args.cheap_llm,
                prompt=prompt_makeup_exam_results.format(
                    scenario.full_record() + scenario.diagnosis_information(),
                    auxiliary_exam_answer,
                ),
                role="makeup exam results",
                ensure_label="answer",
            )
            generated_messages.append(
                {
                    "role": "user",
                    "content": results_auxiliary_exam,
                    "stage": "Auxiliary Exam",
                }
            )
            auxiliary_dialog_str += f"Doctor：{auxiliary_exam_answer}\nResults：{results_auxiliary_exam}\n\n"
    return generated_messages


def run_one_scenario(_scenario_id: int, scenario: Scenario_OSCE, full_ratio=0.25):
    LLM_caller = LLM_Caller_for_One_Thread(
        introduction_log=f"\n<hr>\n\n## {_scenario_id+1}\n",
        dict_from_sys_prompt_to_context_id=global_dict_from_sys_prompt_to_context_id,
    )
    results_dict = None
    try:
        if not scenario.cover_3_stages():
            raise ValueError("病历质量差，未覆盖3个阶段")
        # 考虑到问诊+体检的数据量太多，所以一部分病历跳过，直接辅检
        skip_inquiry = random.uniform(0, 1) > full_ratio
        if not skip_inquiry:
            generated_messages_inquiry = (
                generate_conversation_inquiry_and_physical_exam(scenario, LLM_caller)
            )
            inquiry_cnt = sum(
                [
                    1
                    for m in generated_messages_inquiry
                    if m["stage"] == "Inquiry" and m["role"] == "assistant"
                ]
            )
            physical_cnt = sum(
                [
                    1
                    for m in generated_messages_inquiry
                    if m["stage"] == "Physical Exam" and m["role"] == "assistant"
                ]
            )
            if inquiry_cnt >= 5 and physical_cnt >= 1:
                generated_messages_inquiry_thinking = (
                    generate_thinking_inquiry_and_physical_exam(
                        generated_messages_inquiry, LLM_caller
                    )
                )
                generated_messages_final = generate_auxiliary_exam(
                    scenario,
                    generated_messages_inquiry_thinking,
                    summary_inquiry_physical=generated_messages_inquiry_thinking[
                        -1
                    ].get("summary", ""),
                    LLM_caller=LLM_caller,
                    last_physical_action=generated_messages_inquiry_thinking[-2][
                        "content"
                    ],
                )
            else:
                skip_inquiry = True

        if skip_inquiry:
            # 跳过问诊体检，直接辅检
            record = "病史信息：\n" + scenario.patient_information() + "\n\n"
            record += "体格检查结果：\n" + scenario.physical_exam_information()
            if random.uniform(0, 1) < 0.8:
                summary_inquiry_physical = record
            else:
                summary_inquiry_physical = LLM_caller.query_model_and_extract_label(
                    model_str=args.cheap_llm,
                    prompt=prompt_summary_record.format(record),
                    role="summary from record",
                    ensure_label="answer",
                )
            generated_messages_final = generate_auxiliary_exam(
                scenario,
                [],
                summary_inquiry_physical=summary_inquiry_physical,
                LLM_caller=LLM_caller,
            )
        results_dict = {
            "success": True,
            "scenario_id": _scenario_id,
            "correct_diagnosis": scenario.diagnosis_information(),
            "conversation_str": transform_messages_to_conversation_str(
                generated_messages_final
            ),
            "generated_messages": generated_messages_final,
            "OSCE_Examination": scenario.scenario_dict["OSCE_Examination"],
            # "detailed_log": LLM_caller.LLM_log_list,
        }
    except Exception as e:
        print("Error:", _scenario_id, e, flush=True)
        results_dict = {
            "success": False,
            "scenario_id": _scenario_id,
            "conversation_str": "",
            # "detailed_log": LLM_caller.LLM_log_list,
        }
    return results_dict, _scenario_id


def main():
    scenario_loader = Scenario_OSCE_Loader(args.dataset_path)
    results_list = []
    # 中断后，继续生成
    if args.continue_file_path:
        results_list = read_json(args.continue_file_path)["results_list"]
        print(f"Continue from {args.continue_file_path}", len(results_list))
    scenario_cnt = {}
    for res in results_list:
        if res.get("success", False):
            scenario_cnt[res["scenario_id"]] = (
                scenario_cnt.get(res["scenario_id"], 0) + 1
            )

    if args.num_scenarios is None:
        args.num_scenarios = scenario_loader.num_scenarios
    total_num_scenarios = min(args.num_scenarios, scenario_loader.num_scenarios)

    # 创建进程池并提交任务
    with ProcessPoolExecutor(max_workers=args.parallel_thread_num) as executor:
        future_to_scenario = {}
        for _scenario_id in range(total_num_scenarios):
            # 一个病历，重复多次
            for repeat_times in range(
                scenario_cnt.get(_scenario_id, 0), args.repeat_num
            ):
                future = executor.submit(
                    run_one_scenario,
                    _scenario_id,
                    scenario_loader.get_scenario(_scenario_id),
                )
                future_to_scenario[future] = _scenario_id * 100 + repeat_times

        # 处理完成的每个进程
        for future in as_completed(future_to_scenario):
            # 获取结果
            info_dict, _scenario_id = future.result()
            if info_dict.get("success", False):
                print(f"Scene {_scenario_id+1}: done", flush=True)
                results_list.append(info_dict)
                if len(results_list) % 100 == 0:
                    write_json(
                        {"settings": args.to_dict(), "results_list": results_list},
                        f"temp/{train_id}_{len(results_list)}.json",
                    )
            else:
                print(f"Scene {_scenario_id+1}: skipped")

    results_list = sorted(results_list, key=lambda x: x["scenario_id"])

    for item in results_list:
        print(f"### {item['scenario_id']+1}")
        print(item["conversation_str"])
        del item["conversation_str"]
        # if "detailed_log" in item:
        #     del item["detailed_log"]

    # 平均对话长度
    for res in results_list:
        res["conversation_length"] = len(res.get("generated_messages", []))
    write_json(
        {"settings": args.to_dict(), "results_list": results_list},
        f"detailed_log/{train_id}.json",
    )


def prepare_cache_context_id():
    """
    考虑到prompt cache
    """
    llm_list = [args.cheap_llm, args.expensive_llm]
    for llm in llm_list:
        if llm not in cached_LLM_id_list:
            continue
        cache_system_prompt_list = []
        for pormpt in cache_system_prompt_list:
            global_dict_from_sys_prompt_to_context_id[pormpt] = get_context_id(
                sys_prompt=pormpt,
                model_id=llm,
            )


if __name__ == "__main__":
    prepare_cache_context_id()
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    dataset_id = (
        "-".join([Path(p).stem for p in args.dataset_path])
        .replace("-fill_physical", "")
        .replace("filter-", "")
        .replace("OSCE_", "")
    )
    train_id = (
        f"{current_time}-{dataset_id}--doctor={str(args.expensive_llm).split('/')[-1]}"
    )

    print(train_id)
    main()
