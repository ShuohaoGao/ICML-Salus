from argparse import ArgumentParser
import sys

sys.path.append("../")
from benchmark.prompt_Chinese import prompt_makeup_one_exam_item
from utils.call_LLM import LLM_Caller_for_One_Thread
from utils.tools import extract_label, batch_process_parallel
import json
import concurrent.futures
from utils.io_func import read_json, write_json
from prompts_chinese import *
from tqdm import tqdm
from pathlib import Path
import os

# todo 注意切换expensive_llm，可以试试
config_dict = {
    "expensive_llm": "deepseek-v3-2-251201-enable_thinking",
    # "cheap_llm": "GLM-Z1-Flash",
    # "expensive_llm": "Qwen3-32B",
    # "expensive_llm": "GLM-Z1-Flash",
    "cheap_llm": "deepseek-v3-2-251201",
    # "expensive_llm": "Qwen2.5-32B-Instruct",
    # "cheap_llm": "Qwen2.5-32B-Instruct",
}
physical_index_path = (
    "/data/InteractiveMedLLM/data/OSCE_Chinese_index/index_physical_v2.json"
)
auxiliary_index_path = (
    "/data/InteractiveMedLLM/data/OSCE_Chinese_index/index_auxiliary_v2.json"
)


def judge_one_patient(
    case_study: dict,
    model_name: str,
    try_cnt=5,
):
    """
    case_study:
    {
        "raw_data": {...},
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True/False,
        ...
    }
    判断是否只有一个病人，并且有完整的诊断结果

    如果满足条件，向case_study中添加字段 one_patient=True
    如果不满足条件，向case_study中添加字段 one_patient=False
    """
    assert "raw_data" in case_study, "case_study must contain raw_data field"
    case_study["one_patient"] = False
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_judge_one_patient_and_diagnosis.format(case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if not ans or "none" == ans.lower() or len(ans) == 0 or len(ans) > 200:
                continue
            case_study["one_patient"] = True
            break
        except Exception as e:
            continue
    return case_study


def judge_coverage_3_stages(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True/False,
        ...
    }
    判断是否覆盖3个阶段

    如果满足条件，向case_study中添加字段 coverage_3_stages=True
    如果不满足条件，向case_study中添加字段 coverage_3_stages=False
    """
    if "raw_data" not in case_study or not case_study.get("one_patient", False):
        return None
    case_study["coverage_3_stages"] = True
    return case_study
    case_study["coverage_3_stages"] = False
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_coverage_3_stages.format(case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if ans and ans.lower().startswith("yes"):
                case_study["coverage_3_stages"] = True
                break
        except Exception as e:
            continue
    return case_study


def get_full_diagnosis(
    case_study: dict,
    model_name: str,
    try_cnt=2,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        ...
    }
    提取完整的诊断结果 向case_study中添加字段 core_diagnosis, full_diagnosis
    """
    if (
        "raw_data" not in case_study
        or not case_study.get("one_patient", False)
        or not case_study.get("coverage_3_stages", False)
    ):
        return None
    LLM_caller = LLM_Caller_for_One_Thread()
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    case_study["core_diagnosis"] = None
    case_study["full_diagnosis"] = None
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model(
                model_str=model_name,
                prompt=prompt_get_full_diagnosis.format(case_study_str),
                system_prompt=None,
                ensure_label="核心诊断",
            )
            core_diagnosis = extract_label(ans, label="核心诊断")
            if (
                not core_diagnosis
                or "none" == core_diagnosis.lower()
                or len(core_diagnosis) == 0
                or len(core_diagnosis) > 200
            ):
                continue
            full_diagnosis = extract_label(ans, label="完整诊断")
            if (
                not full_diagnosis
                or "none" == full_diagnosis.lower()
                or len(full_diagnosis) == 0
                or len(full_diagnosis) > 2000
            ):
                continue
            case_study["core_diagnosis"] = core_diagnosis
            case_study["full_diagnosis"] = full_diagnosis
            break
        except Exception as e:
            continue
    if "OSCE_Examination" not in case_study:
        case_study["OSCE_Examination"] = {}
    case_study["Correct_Diagnosis"] = (
        f"核心诊断: {case_study['core_diagnosis']}\n完整诊断: {case_study['full_diagnosis']}"
    )
    case_study["OSCE_Examination"]["Correct_Diagnosis"] = case_study[
        "Correct_Diagnosis"
    ]
    case_study["OSCE_Examination"]["core_diagnosis"] = case_study["core_diagnosis"]
    case_study["OSCE_Examination"]["full_diagnosis"] = case_study["full_diagnosis"]
    return case_study


def judge_enough_data_for_diagnosis(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        ...
    }
    判断是否包含足够的诊断信息，且诊断完整、不矛盾


    或者
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "OSCE_Examination": {...}
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "OSCE_Examination": {...}
        "enough_data_extract": True/False,
        ...
    }
    """
    if (
        "raw_data" not in case_study
        or case_study.get("core_diagnosis", None) is None
        or case_study.get("full_diagnosis", None) is None
    ):
        return None
    LLM_caller = LLM_Caller_for_One_Thread()
    if (
        "OSCE_Examination" not in case_study
        or "Test_Results" not in case_study["OSCE_Examination"]
    ):
        case_study_str = json.dumps(
            case_study["raw_data"], ensure_ascii=False, indent=2
        )
        key_name = "enough_data_raw"
    else:
        case_study_str = json.dumps(
            case_study["OSCE_Examination"], ensure_ascii=False, indent=2
        )
        key_name = "enough_data_extract"
    diagnosis_str = case_study["Correct_Diagnosis"]
    case_study[key_name] = False
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_judge_enough_data.format(diagnosis_str, case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if ans and ans.lower().startswith("yes"):
                case_study[key_name] = True
                break
        except Exception as e:
            continue
    return case_study


def extract_HPI_dict(
    case_study: dict,
    model_name: str,
    try_cnt=2,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "OSCE_Examination": {
            ""Patient_Actor": {...}
        }
        ...
    }
    提取病史，并返回字典格式
    """
    if (
        "raw_data" not in case_study
        or not case_study.get("one_patient", False)
        or not case_study.get("coverage_3_stages", False)
        or case_study.get("full_diagnosis", None) is None
    ):
        return None
    LLM_caller = LLM_Caller_for_One_Thread()
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    if "OSCE_Examination" not in case_study:
        case_study["OSCE_Examination"] = {}
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model(
                model_str=model_name,
                prompt=prompt_HPI.format(case_study_str),
                system_prompt=None,
                ensure_label=None,
            )
            if "```json" in ans:
                ans = ans.split("```json")[-1].split("```")[0].strip()
            res = json.loads(ans)
            if isinstance(res, dict) and len(res) > 0:
                case_study["OSCE_Examination"]["Patient_Actor"] = res
                break
        except Exception as e:
            continue
    if "Patient_Actor" not in case_study["OSCE_Examination"]:
        return None
    if not case_study["OSCE_Examination"]["Patient_Actor"].get("主诉", ""):
        return None
    first_sentence = (
        "基本信息："
        + case_study["OSCE_Examination"]["Patient_Actor"].get("基本信息", "无。")
        + "\n"
        + "主诉："
    )
    first_sentence += case_study["OSCE_Examination"]["Patient_Actor"].get("主诉", "")
    case_study["OSCE_Examination"]["Objective_for_Doctor"] = first_sentence
    return case_study


def extract_physical_and_auxiliary(
    case_study: dict,
    model_name: str,
    str_len_threshold=3000,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "体格检查": "...",
        "辅助检查": "...",
        ...
    }

    提取出体格检查和辅助检查的文本
    """
    if "raw_data" not in case_study:
        return None

    def parse(case_study_str: str):
        LLM_caller = LLM_Caller_for_One_Thread()
        for _ in range(3):
            try:
                ans = LLM_caller.query_model(
                    model_str=model_name,
                    prompt=prompt_all_examination.format(case_study_str),
                    system_prompt=None,
                    ensure_label=None,
                )
                if "```json" in ans:
                    ans = ans.split("```json")[-1].split("```")[0].strip()
                res = json.loads(ans)
                if len(res) == 0:
                    continue
                if "体格检查" not in res:
                    res["体格检查"] = ""
                if "辅助检查" not in res:
                    res["辅助检查"] = ""
                return res
            except Exception as e:
                continue
        return {"体格检查": "", "辅助检查": ""}

    raw_case_study = case_study["raw_data"]
    if isinstance(raw_case_study, dict):
        # 拆分成多个部分，避免丢失信息
        res_dict = {"体格检查": "", "辅助检查": ""}
        case_study_str = ""
        cnt = 0
        for idx, (k, v) in enumerate(raw_case_study.items()):
            case_study_str += f"{k}: {v}\n\n"
            if (
                len(case_study_str) > str_len_threshold
                or idx == len(raw_case_study) - 1
            ):
                temp_dict = parse(case_study_str=case_study_str)
                if temp_dict["体格检查"]:
                    res_dict["体格检查"] += str(temp_dict["体格检查"]) + "\n"
                if temp_dict["辅助检查"]:
                    res_dict["辅助检查"] += str(temp_dict["辅助检查"]) + "\n"
                case_study_str = ""
                cnt += 1
        if cnt > 1:
            # 重新合并
            res_dict = parse(
                case_study_str=json.dumps(res_dict, ensure_ascii=False, indent=2)
            )
    else:
        res_dict = parse(case_study_str=str(raw_case_study))
    case_study.update(res_dict)
    return case_study


def get_all_leaves_with_path(data):
    """
    从一个多级嵌套索引中，获取所有叶子节点的路径（tuple 形式）

    输入：
    {
        "X线检查": {
            "胸部": [
                "胸部X线摄影"
            ],
            "骨骼关节": [
                "脊柱X线摄影",
                "四肢骨X线摄影"
            ]
        }
    }

    输出：
    ('X线检查', '胸部', '胸部X线摄影')
    ('X线检查', '骨骼关节', '脊柱X线摄影')
    ('X线检查', '骨骼关节', '四肢骨X线摄影')
    """
    result = []

    def recursive_extract(current, path):
        if isinstance(current, dict):
            for key, value in current.items():
                recursive_extract(value, path + (key,))
        elif isinstance(current, list):
            for item in current:
                recursive_extract(item, path)
        else:
            result.append(path + (current,))

    recursive_extract(data, ())
    return result


def dfs_classify(
    case_study_str: str,
    tree_dict: dict,
    model_name: str,
    LLM_caller: LLM_Caller_for_One_Thread,
):
    # 递归终止条件
    leaves_key_list = get_all_leaves_with_path(tree_dict)
    if (
        len(leaves_key_list) < 10 and len(leaves_key_list) == len(set(leaves_key_list))
    ) or isinstance(tree_dict, list):
        key_list = [path_tuple[-1] for path_tuple in leaves_key_list] + ["其他"]
        for _ in range(3):
            try:
                ans = LLM_caller.query_model_and_extract_label(
                    model_str=model_name,
                    prompt=USER_prompt_all.format(case_study_str),
                    system_prompt=SYSTEM_prompt_classify.format(key_list),
                    ensure_label="answer",
                )
                if "```json" in ans:
                    ans = ans.split("```json")[1].split("```")[0].strip()
                res = json.loads(ans)
                # 检查格式
                ok = True
                for k in res:
                    if k not in key_list:
                        ok = False
                    elif not isinstance(res[k], str):
                        ok = False
                if not ok:
                    continue
                # 还原为多级索引
                final_res = {}
                for path_tuple in leaves_key_list:
                    if path_tuple[-1] not in res:
                        continue
                    current_level = final_res

                    for key in path_tuple[:-1]:  # 所有中间层级
                        if key not in current_level:
                            current_level[key] = {}
                        current_level = current_level[key]

                    # 设置最终叶子节点的值
                    current_level[path_tuple[-1]] = res[path_tuple[-1]]
                if "其他" in res:
                    final_res["其他"] = res["其他"]
                return final_res
            except Exception as e:
                continue

    # 递归
    key_list = [k for k in tree_dict] + ["其他"]
    for _ in range(3):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=USER_prompt_all.format(case_study_str),
                system_prompt=SYSTEM_prompt_classify.format(str(key_list)),
                ensure_label="answer",
            )
            if "```json" in ans:
                ans = ans.split("```json")[1].split("```")[0].strip()
            res = json.loads(ans)
            # 检查格式
            ok = True
            for k in res:
                if k not in key_list:
                    ok = False
                elif not isinstance(res[k], str):
                    ok = False
            if not ok:
                continue
            # dfs递归分类
            for k in res:
                if k == "其他":
                    continue
                res[k] = dfs_classify(
                    res[k], tree_dict[k], model_name=model_name, LLM_caller=LLM_caller
                )
            return res
        except Exception as e:
            continue
    return case_study_str


def classify_physical_index(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "体格检查": "...",
        "辅助检查": "...",
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "体格检查": "...",
        "辅助检查": "...",
        "OSCE_Examination": {
            ""Physical_Examination_Findings": {...}
        }
        ...
    }
    """
    if "体格检查" not in case_study:
        return None
    if "OSCE_Examination" not in case_study:
        case_study["OSCE_Examination"] = {}
    physical_index = read_json(physical_index_path)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = dfs_classify(
                case_study_str=case_study["体格检查"],
                tree_dict=physical_index,
                model_name=model_name,
                LLM_caller=LLM_caller,
            )
            if isinstance(ans, str):
                ans = {"其他": ans}
            case_study["OSCE_Examination"]["Physical_Examination_Findings"] = ans
            break
        except Exception as e:
            continue
    if "Physical_Examination_Findings" not in case_study["OSCE_Examination"]:
        return None
    return case_study


def classify_auxiliary_index(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "体格检查": "...",
        "辅助检查": "...",
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "体格检查": "...",
        "辅助检查": "...",
        "OSCE_Examination": {
            ""Test_Results": {...}
        }
        ...
    }
    """
    if "辅助检查" not in case_study:
        return None
    if "OSCE_Examination" not in case_study:
        case_study["OSCE_Examination"] = {}
    auxiliary_index = read_json(auxiliary_index_path)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = dfs_classify(
                case_study_str=case_study["辅助检查"],
                tree_dict=auxiliary_index,
                model_name=model_name,
                LLM_caller=LLM_caller,
            )
            if isinstance(ans, str):
                ans = {"其他": ans}
            case_study["OSCE_Examination"]["Test_Results"] = ans
            break
        except Exception as e:
            continue
    if "Test_Results" not in case_study["OSCE_Examination"]:
        return None
    return case_study


def extract_difficulty_level(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "OSCE_Examination": {
            ...
        }
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "OSCE_Examination": {
            ...
        },
        "difficulty" : "Level 1"/"Level 2"/"Level 3"/"Level 4"/None,
        ...
    }
    提取诊断难度等级
    """
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Physical_Examination_Findings" not in case_study["OSCE_Examination"]
        or "Test_Results" not in case_study["OSCE_Examination"]
    ):
        return None
    LLM_caller = LLM_Caller_for_One_Thread()
    case_study_str = json.dumps(
        case_study["OSCE_Examination"], ensure_ascii=False, indent=2
    )
    case_study["difficulty"] = None
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_difficulty_level.format(case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if ans.lower().startswith("level") and len(ans) < 50:
                case_study["difficulty"] = ans
                break
        except Exception as e:
            continue
    return case_study


def rewrite_chief_complaint(
    case_study: dict,
    model_name: str,
    try_cnt=2,
):
    """
    case_study:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "OSCE_Examination": {
            "Objective_for_Doctor": "改写前的主诉",
            ...
        }
        ...
    }

    return:
    {
        "raw_data": {...},
        "one_patient": True,
        "coverage_3_stages": True,
        "core_diagnosis": "xxx" / None,
        "full_diagnosis": "xxx" / None,
        "enough_data_raw": True/False,
        "OSCE_Examination": {
            "Objective_for_Doctor": "改写后的主诉，第一人称",
            ...
        },
        ...
    }
    改写其中的主诉
    """
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
    ):
        return None
    record_info = case_study["OSCE_Examination"]["Patient_Actor"]
    if "基本信息" in record_info:
        record_info["基本信息"] = record_info["基本信息"].replace("\n", " ")
    if "主诉" in record_info:
        del record_info["主诉"]
    LLM_caller = LLM_Caller_for_One_Thread()
    chief_complaint = ""  # 选择最长的一次
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_rewrite_Zhusu.format(record_info),
                system_prompt=None,
                ensure_label="answer",
            )
            if ans and len(ans) < 1000 and len(ans) > len(chief_complaint):
                chief_complaint = ans
        except Exception as e:
            continue
    first_sentence = (
        "基本信息："
        + case_study["OSCE_Examination"]["Patient_Actor"].get("基本信息", "无。")
        + "\n"
        + "主诉："
    )
    first_sentence += chief_complaint
    case_study["OSCE_Examination"]["Objective_for_Doctor"] = first_sentence
    record_info["主诉"] = chief_complaint
    case_study["OSCE_Examination"]["Patient_Actor"] = record_info
    return case_study


def verify_validness(data_dict: dict):
    key_list = ["difficulty", "OSCE_Examination", "departments_PUMCH"]
    for key in key_list:
        if key not in data_dict:
            return False
    key_list_in_OSCE = [
        "Patient_Actor",
        "Physical_Examination_Findings",
        "Test_Results",
        "Correct_Diagnosis",
        "Objective_for_Doctor",
    ]
    for key in key_list_in_OSCE:
        if key not in data_dict["OSCE_Examination"]:
            return False
        if len(data_dict["OSCE_Examination"][key]) == 0:
            return False
    return True


def judge_department(
    patient_info: str,
    department_cls: str,
    task_id: int,
    model_name="GLM-Z1-Flash",
):
    """判断病人是否属于某个科室"""
    try:
        sys_prompt = SYSTEM_prompt_department_class
        user_prompt = USER_prompt_department_class.format(patient_info, department_cls)
        LLM_caller = LLM_Caller_for_One_Thread()
        cnt = 0
        try_cnt = 3
        for _ in range(try_cnt):
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=user_prompt,
                system_prompt=sys_prompt,
                ensure_label="answer",
            )
            if ans and ans.lower().startswith("yes"):
                cnt += 1
                if cnt * 2 >= try_cnt:
                    break
        judge = cnt * 2 >= try_cnt
        res_dict = {
            "task_id": task_id,
            "department_cls": department_cls,
            "judge": judge,
        }
        return res_dict
    except:
        return None


def parallel_annotate_department(
    data_list: list, model_name="GLM-Z1-Flash", max_workers=300
):
    """
    并行标注所属科室
    对每一份病历标注科室，允许同时属于多个科室

    参数:
        data_list: 包含OSCE_Examination字段的数据列表

    返回:
        新增departments_PUMCH字段的病例列表
        在原有的每个dict中，新加一个属性：
        {
            ...
            "departments_PUMCH": [...]
        }
    """
    global department_list
    idx_to_case_study = {}
    for idx, item in enumerate(data_list):
        case_study = item.copy()
        case_study["departments_PUMCH"] = []
        idx_to_case_study[idx] = case_study
    results_list = []
    model_name = config_dict["cheap_llm"]
    # 创建进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, case_study in enumerate(data_list):
            for department_cls in department_list:
                case_study_str = ""
                if "OSCE_Examination" in case_study:
                    case_study_str = json.dumps(
                        case_study["OSCE_Examination"], ensure_ascii=False, indent=2
                    )
                else:
                    case_study_str = json.dumps(
                        case_study, ensure_ascii=False, indent=2
                    )
                futures.append(
                    executor.submit(
                        judge_department,
                        case_study_str,
                        department_cls,
                        idx,
                        model_name,
                    )
                )

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(data_list) * len(department_list),
            desc="Processing",
        ):
            case_res = future.result()
            if case_res is None:
                print("skipped")
                continue
            results_list.append(case_res)
    for res in results_list:
        idx = res["task_id"]
        if res.get("judge", False):
            idx_to_case_study[idx]["departments_PUMCH"].append(res["department_cls"])
    final_results_list = []
    for idx, case_study in idx_to_case_study.items():
        case_study["departments_PUMCH"] = list(set(case_study["departments_PUMCH"]))
        final_results_list.append(case_study)
    return final_results_list


def get_item_name_based_on_text(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    {
        "其他": "眼底检查显示双眼视盘肿胀、周边视网膜出血，右眼还有血管周围渗出物"
    }
    改写为
    {
        "其他: 眼底检查": "眼底检查显示双眼视盘肿胀、周边视网膜出血，右眼还有血管周围渗出物"
    }
    """
    LLM_caller = LLM_Caller_for_One_Thread()
    item_type = "体格检查"  # 体检 or 辅检

    def modify_other_values(nested_dict):
        # 使用列表收集需要修改的路径
        to_modify = []

        stack = [nested_dict]  # 栈中存储当前层的字典
        while stack:
            current_dict = stack.pop()
            for key, value in current_dict.items():
                if key == "其他":
                    # 记录需要修改的路径和值
                    to_modify.append((current_dict, value))
                elif isinstance(value, dict):
                    stack.append(value)

        for parent_dict, value in to_modify:
            if "其他" in parent_dict:
                for _ in range(try_cnt):
                    try:
                        ans = LLM_caller.query_model_and_extract_label(
                            model_str=model_name,
                            prompt=USER_prompt_all.format(f"{item_type}: {value}"),
                            system_prompt=SYSTEM_prompt_get_item_name,
                            ensure_label="answer",
                        )
                        if (
                            not ans
                            or "none" == ans.lower()
                            or len(ans) == 0
                            or len(ans) > 200
                        ):
                            continue
                        new_key = f"其他: {ans}"
                        del parent_dict["其他"]
                        parent_dict[new_key] = value
                        break
                    except Exception as e:
                        continue

    # try:
    #     nested_dict = case_study["OSCE_Examination"]["Physical_Examination_Findings"]
    #     item_type = "体格检查"
    #     modify_other_values(nested_dict)
    # except:
    #     pass
    try:
        nested_dict = case_study["OSCE_Examination"]["Test_Results"]
        item_type = "辅助检查"
        modify_other_values(nested_dict)
    except:
        pass
    return case_study


def makeup_physical_exam_results_within_group(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    根据系统部位分组，进行体检的补全
    """
    if (
        "OSCE_Examination" not in case_study
        or "Physical_Examination_Findings" not in case_study["OSCE_Examination"]
    ):
        return None
    physical_index = read_json(physical_index_path)
    LLM_caller = LLM_Caller_for_One_Thread()
    stack = [
        (
            physical_index,
            case_study["OSCE_Examination"]["Physical_Examination_Findings"],
        )
    ]  # 栈中存储当前层的字典
    patient_record = json.dumps(
        case_study["OSCE_Examination"], ensure_ascii=False, indent=2
    )
    while stack:
        current_index, current_data_dict = stack.pop()
        for key, value in current_index.items():
            if key not in current_data_dict:
                continue
            if isinstance(value, list):
                need_makeup_items = []
                for exam_item in value:
                    if exam_item not in current_data_dict[key]:
                        need_makeup_items.append(exam_item)
                if len(need_makeup_items) > 0:
                    need_makeup_items_str = "\n".join(need_makeup_items)
                    for _ in range(try_cnt):
                        try:
                            ans = LLM_caller.query_model_and_extract_label(
                                model_str=model_name,
                                prompt=USER_prompt_makeup_results.format(
                                    patient_record, need_makeup_items_str
                                ),
                                system_prompt=SYSTEM_prompt_makeup_results,
                                ensure_label="answer",
                            )
                            makeup_data_dict = json.loads(ans)
                            # 检查需要编造的项目是否都在
                            if not all(
                                [
                                    item_name in makeup_data_dict
                                    for item_name in need_makeup_items
                                ]
                            ):
                                continue
                            # 把编造的结果填入data中
                            for item_name in need_makeup_items:
                                current_data_dict[key][item_name] = makeup_data_dict[
                                    item_name
                                ]
                            break
                        except Exception as e:
                            continue
            elif isinstance(value, dict):
                stack.append((value, current_data_dict[key]))
    return case_study


def check_item_results(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    逐项判断，检查结果是否是完整的
    """
    LLM_caller = LLM_Caller_for_One_Thread()

    def handle_one_item(item_name: str, item_result: str, raw_text: str):
        # 检查结果是否完整
        for _ in range(try_cnt):
            res = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=USER_prompt_check_WanZheng.format(item_name, item_result),
                system_prompt=SYSTEM_prompt_check_WanZheng,
                ensure_label="answer",
            )
            if res and res.lower().startswith("yes"):
                return item_result
        # 不完整，重新提取
        res = LLM_caller.query_model_and_extract_label(
            model_str=model_name,
            prompt=USER_prompt_extract_item_result.format(item_name, raw_text),
            system_prompt=SYSTEM_prompt_extract_item_result,
            ensure_label="answer",
        )
        if not res or "none" == res.lower() or len(res) > 800:
            return ""
        # print(item_result, '----->', res)
        return res

    def dfs_handle_dict(current_dict: dict, raw_text: str):
        for key, value in current_dict.items():
            if isinstance(value, str) and not key.startswith("其他"):
                current_dict[key] = handle_one_item(
                    item_name=key, item_result=value, raw_text=raw_text
                )
            elif isinstance(value, dict):
                dfs_handle_dict(value, raw_text)
        # 检查空的dict或者item
        key_list = [k for k in current_dict]
        for k in key_list:
            if len(current_dict[k]) == 0:
                del current_dict[k]

    # try:
    #     dfs_handle_dict(
    #         current_dict=case_study["OSCE_Examination"][
    #             "Physical_Examination_Findings"
    #         ],
    #         raw_text=case_study["体格检查"],
    #     )
    # except:
    #     pass
    try:
        dfs_handle_dict(
            current_dict=case_study["OSCE_Examination"]["Test_Results"],
            raw_text=case_study["辅助检查"],
        )
    except:
        pass
    return case_study


def get_list_group(d: dict) -> dict:
    """
    得到若干key=组名，value=检查项目list
    """

    def dfs_get_group_name(current_dict: dict) -> dict:
        res = {}
        for key, value in current_dict.items():
            if isinstance(value, list):
                res[key] = value
            elif isinstance(value, dict):
                res.update(dfs_get_group_name(value))
        return res

    return dfs_get_group_name(d)


def makeup_necessary_physical_exam_results(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    def dfs_put_results(current_dict: dict, group_item_results_dict: dict):
        d = {}
        for k, v in current_dict.items():
            if k in group_item_results_dict:
                d[k] = group_item_results_dict[k]
            elif isinstance(v, dict):
                temp = dfs_put_results(v, group_item_results_dict)
                if len(temp) > 0:
                    d[k] = temp
        return d

    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Correct_Diagnosis" not in case_study["OSCE_Examination"]
        or "体格检查" not in case_study
        or "辅助检查" not in case_study
        # or len(case_study["体格检查"]) < 50
        # or len(case_study["辅助检查"]) < 20
    ):
        return None
    try:
        LLM_caller = LLM_Caller_for_One_Thread()
        record_inquiry = json.dumps(
            case_study["OSCE_Examination"]["Patient_Actor"],
            indent=2,
            ensure_ascii=False,
        )
        diagnosis = case_study["OSCE_Examination"]["Correct_Diagnosis"]
        physical_index_dict = read_json(physical_index_path)
        group_itemList_dict = get_list_group(physical_index_dict)
        group_itemList_dict_str = json.dumps(
            group_itemList_dict, indent=2, ensure_ascii=False
        )
        # 获取必要的体格检查的组名
        group_list = []
        for _ in range(try_cnt):
            try:
                res = LLM_caller.query_model_and_extract_label(
                    model_str=model_name,
                    prompt=prompt_get_necessary_physical_groups.format(
                        record_inquiry, diagnosis, group_itemList_dict_str
                    ),
                    system_prompt=None,
                    ensure_label="answer",
                )
                group_list = [x.strip() for x in res.split("\n") if x.strip()]
                if all([group in group_itemList_dict for group in group_list]):
                    break
            except:
                continue
        if not group_list or not all(
            [group in group_itemList_dict for group in group_list]
        ):
            return None
        # 按照分组，分别补全体检
        full_record = (
            f"病史信息:\n{record_inquiry}\n"
            + f"部分体格检查结果:\n{case_study['体格检查']}\n"
            + f"辅助检查结果:\n{case_study['辅助检查']}\n\n"
            + diagnosis
        )
        group_item_results_dict = {}
        for group in group_list:
            item_list = group_itemList_dict[group]
            # 编造这一组检查
            res_dict = None
            for _ in range(try_cnt):
                try:
                    res = LLM_caller.query_model_and_extract_label(
                        model_str=model_name,
                        prompt=USER_prompt_makeup_results.format(
                            full_record, "\n".join(item_list)
                        ),
                        system_prompt=SYSTEM_prompt_makeup_results,
                        ensure_label="answer",
                    )
                    d = json.loads(res)
                    if all([item in item_list for item in d]):
                        res_dict = d
                        break
                except:
                    continue
            if res_dict and isinstance(res_dict, dict):
                group_item_results_dict[group] = res_dict

        # 得到了补全的结果，dfs更新到体检索引中
        OSCE_physical_res = dfs_put_results(
            physical_index_dict, group_item_results_dict
        )
        if isinstance(OSCE_physical_res, dict) and len(OSCE_physical_res) > 0:
            case_study["OSCE_Examination"][
                "Physical_Examination_Findings"
            ] = OSCE_physical_res
            return case_study
        else:
            return None
    except:
        return None


def judge_misdiagnosis_and_doctor(
    case_study: dict,
    model_name: str,
):
    """
    case_study:
    {
        "raw_data": {...},
        ...
    }

    return:
    {
        "raw_data": {...},
        "misdiagnosis": "误诊诊断"/None,
        ...
    }
    判断是否误诊，然后提取出对应的Doctor
    """
    assert "raw_data" in case_study, "case_study must contain raw_data field"
    case_study["misdiagnosis"] = None
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    LLM_caller = LLM_Caller_for_One_Thread()
    try:
        ans = LLM_caller.query_model_and_extract_label(
            model_str=model_name,
            prompt=prompt_judge_misdiagnosis_strict.format(case_study_str),
            system_prompt=None,
            ensure_label="answer",
        )
        if ans and 0 < len(ans) < 1000:
            lines = [x.strip() for x in ans.split("\n") if x.strip()]
            if len(lines) == 3:
                mis_diagnosis = lines[0]
                mis_doctor = lines[1]
                final_doctor = lines[2]
                if "none" != mis_diagnosis.lower():
                    case_study["misdiagnosis"] = mis_diagnosis
                    case_study["bad_doctor"] = (
                        mis_doctor if mis_doctor.lower() != "none" else None
                    )
                    case_study["good_doctor"] = (
                        final_doctor if final_doctor.lower() != "none" else None
                    )
    except Exception as e:
        pass
    return case_study


def judge_location(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    """
    case_study:
    {
        "raw_data": {...},
        ...
    }

    return:
    {
        "raw_data": {...},
        "location": "北京"/None,
        ...
    }
    判断位置信息，用于多中心
    """
    assert "raw_data" in case_study, "case_study must contain raw_data field"
    case_study["location"] = None
    if "source" in case_study and case_study["source"].startswith("iiy"):
        case_study["location"] = "中国"
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_get_location.format(case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if not ans or "none" == ans.lower() or len(ans) == 0 or len(ans) > 200:
                continue
            case_study["location"] = ans
            break
        except Exception as e:
            continue
    return case_study


def classify_research_department(
    case_study: dict,
    model_name: str,
    try_cnt=2,
    department_list=[
        "肿瘤科",
        "心血管内科",
        "神经内科",
        "重症医学科",
        "老年科",
        "呼吸科",
        "内分泌科",
        "肾脏内科",
        "风湿免疫科",
        "消化内科",
        "妇产科",
        "普通外科",
        "泌尿外科",
        "骨科",
    ],
):
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Correct_Diagnosis" not in case_study["OSCE_Examination"]
    ):
        return None
    hpi = case_study["OSCE_Examination"]["Patient_Actor"]
    diagnosis = case_study["OSCE_Examination"]["Correct_Diagnosis"]
    info = f"病史信息：\n{hpi}\n\n{diagnosis}"
    key = f"research_department_{model_name}"
    case_study[key] = "其他"
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_classify_research_department.format(
                    ", ".join(department_list), info
                ),
                system_prompt=None,
                ensure_label="answer",
            )
            if not ans or len(ans) == 0 or len(ans) > 200:
                continue
            if ans not in department_list and not ans.startswith("其他"):
                continue
            case_study[key] = ans
            break
        except Exception as e:
            continue
    return case_study


def verify_quality(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    case_study_str = json.dumps(case_study["raw_data"], ensure_ascii=False, indent=2)
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_get_location.format(case_study_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if not ans or "none" == ans.lower() or len(ans) == 0 or len(ans) > 200:
                continue
            case_study["location"] = ans
            break
        except Exception as e:
            continue
    return case_study


def makeup_HPI(
    case_study: dict,
    model_name: str,
    try_cnt=3,
):
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Correct_Diagnosis" not in case_study["OSCE_Examination"]
        or "体格检查" not in case_study
        or "辅助检查" not in case_study
    ):
        return None
    record_str = f"病史信息：\n{case_study['OSCE_Examination']['Patient_Actor']}\n\n"
    record_str += f"体格检查：\n{case_study['体格检查']}\n\n"
    record_str += f"辅助检查：\n{case_study['辅助检查']}\n\n"
    record_str += case_study["OSCE_Examination"]["Correct_Diagnosis"]
    LLM_caller = LLM_Caller_for_One_Thread()
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_makeup_HPI.format(record_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if not ans or "none" == ans.lower() or len(ans) == 0 or len(ans) > 2000:
                continue
            case_study["OSCE_Examination"]["Patient_Actor"]["现病史"] = ans
            break
        except Exception as e:
            continue
    return case_study


def makeup_all_physical_exam_results(
    case_study: dict,
    model_name: str,
    try_cnt=2,
):
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Correct_Diagnosis" not in case_study["OSCE_Examination"]
        or "体格检查" not in case_study
        or "辅助检查" not in case_study
    ):
        return None
    try:
        LLM_caller = LLM_Caller_for_One_Thread()
        record_inquiry = json.dumps(
            case_study["OSCE_Examination"]["Patient_Actor"],
            indent=2,
            ensure_ascii=False,
        )
        diagnosis = case_study["OSCE_Examination"]["Correct_Diagnosis"]
        physical_index_dict = read_json(physical_index_path)
        group_itemList_dict = get_list_group(physical_index_dict)
        # 获取所有的体格检查的组名
        group_list = [k for k in group_itemList_dict]
        # 按照分组，分别补全体检
        full_record = (
            f"病史信息:\n{record_inquiry}\n"
            + f"部分体格检查结果:\n{case_study['体格检查']}\n"
            + f"辅助检查结果:\n{case_study['辅助检查']}\n\n"
            + diagnosis
        )
        final_physical_res = {}
        for group in group_list:
            item_list = group_itemList_dict[group]
            for _ in range(try_cnt):
                try:
                    res = LLM_caller.query_model_and_extract_label(
                        model_str=model_name,
                        prompt=prompt_makeup_physical_exam.format(
                            full_record, "\n".join(item_list)
                        ),
                        system_prompt=None,
                        ensure_label="answer",
                    )
                    final_physical_res[group] = res
                    break
                except:
                    continue
        case_study["OSCE_Examination"][
            "Physical_Examination_Findings"
        ] = final_physical_res
        return case_study
    except:
        return None


def map_physical_name(case_study: dict, model_name: str = ""):
    name_dict = {
        "一般状况与生命体征（意识状态、体温、脉搏、呼吸、血压、发育营养等）": "一般状况与生命体征（意识状态、生命体征、整体外观、身高体重）",
        "皮肤与黏膜系统（颜色、皮疹、皮下结节、黏膜病变、水肿等）": "皮肤与黏膜系统（皮肤颜色、皮疹病变、皮下结节、水肿、皮温湿度、黏膜）",
        "浅表淋巴结检查（头颈/腋窝/腹股沟淋巴结等）": "浅表淋巴结检查（头颈、腋窝、腹股沟淋巴结）",
        "头颅与面部": "头颅与面部（头颅检查、面部检查）",
        "眼部": "眼部检查（视功能、眼外部、眼球）",
        "耳鼻咽喉": "耳鼻咽喉检查（耳部、鼻部、口咽部）",
        "颈部与甲状腺（包括气管、甲状腺等检查）": "颈部与甲周腺（颈部外形活动度、甲状腺、颈部血管、气管）",
        "胸部视触叩听": "胸部视触叩听（视诊、听诊、触诊、叩诊）",
        "心脏视触叩听": "心脏视触叩听（视诊、叩诊、听诊、触诊）",
        "腹部检查（包含肝脾等检查）": "腹部检查（视诊、触诊、叩诊、听诊、肝脾、肾区膀胱）",
        "脊柱检查": "脊柱检查（外形、触诊、活动度）",
        "关节检查": "关节检查（红肿、压痛、活动度、畸形）",
        "四肢检查": "四肢检查（上肢、下肢）",
        "脑神经": "脑神经功能（12对脑神经）",
        "运动系统检查（肌力、肌张力及异常运动观察）": "运动系统检查（肌力、肌张力、肌肉萎缩、不自主运动）",
        "感觉系统（痛觉、触觉、运动觉等）": "感觉系统检查（浅感觉、深感觉、复合感觉）",
        "反射功能检查（深反射、浅反射及病理反射）": "反射功能检查（深反射、浅反射、病理反射）",
        "协调和平衡功能（协调性及步态评估）": "协调和平衡功能（平衡协调、步态分析）",
        "男性生殖器检查（阴茎、阴囊、睾丸及附属结构）": "男性生殖器检查（外生殖器、腹股沟、前列腺）",
        "女性生殖器检查（外阴、阴道、宫颈、子宫及附件）": "女性生殖器检查（外阴尿道、阴道宫颈、子宫附件）",
        "肛门与直肠检查（外观、指诊及前列腺触诊）": "肛门与直肠检查（肛周、直肠指诊）",
        "外周血管检查（动脉搏动、毛细血管充盈、静脉曲张评估等）": "外周血管检查（动脉、静脉、末梢循环）",
        "妇产科（妊娠相关体征及胎儿监测）": "妇产科检查（妊娠期、乳房）",
        "儿科（发育指标及常见异常筛查）": "儿科检查（生长发育、新生儿专项）",
        "老年人检查（跌倒风险及认知状态评估）": "老年人检查（功能评估、风险筛查）",
    }
    old_dict = case_study["OSCE_Examination"]["Physical_Examination_Findings"].copy()
    new_dict = old_dict.copy()
    for key in old_dict:
        if key in name_dict:
            del new_dict[key]
            new_dict[name_dict[key]] = old_dict[key]
    case_study["OSCE_Examination"]["Physical_Examination_Findings"] = new_dict
    return case_study


def makeup_common_auxiliary_results(
    case_study: dict,
    model_name: str,
    auxiliary_exam_list: list = [
        "血常规检查",
        "尿常规检查",
        "粪便常规",
        "粪便隐血试验（OB）（免疫法）",
        "肝功能全套",
        "肾功能全套",
        "电解质测定",
        "血糖测定",
        "血脂全套",
        "心肌酶谱",
        "凝血功能全套",
        "甲状腺功能",
        "肿瘤标志物筛查",
        "骨密度测定",
        "常规心电图费检查",
        "心脏彩色多普勒超声",
        "颈部血管彩色多普勒超声（颈动脉）",
        "甲状腺及颈部淋巴结彩超",
        "彩色多普勒超声常规检查（腹部）",
        "彩色多普勒超声常规检查（胸部）",
        "彩色多普勒超声常规检查（妇科）",
        "四肢血管彩色多普勒超声",
        "冠状动脉CTA",
        "胸部X线正侧位摄影",
        "乳腺钼靶摄片18×24吋",
        "头颅CT平扫+增强",
        "胸部CT平扫+增强",
        "全腹部CT平扫+增强",
        "头颅MRI磁共振平扫+增强",
        "全脊柱磁共振平扫",
        "胃镜（纤维胃十二指肠镜检查）",
        "肠镜（纤维结肠镜检查）",
        "糖化血红蛋白测定（色谱法）",
        "C—反应蛋白测定（CRP）（各种免疫学方法）",
        "血同型半胱氨酸测定（各种免疫学方法）",
        "贫血三项",
        "总前列腺特异性抗原测定（TPSA）（各种免疫学方法）",
        "胃泌素—17检测",
        "血清铁蛋白测定（各种免疫学方法）",
        "乙肝两对半",
        "丙型肝炎抗体测定",
        "HIV抗体测定",
        "人乳头瘤病毒（HPV）核酸检测",
        "幽门螺杆菌抗体测定",
    ],
    try_cnt=3,
):
    if (
        "OSCE_Examination" not in case_study
        or "Patient_Actor" not in case_study["OSCE_Examination"]
        or "Correct_Diagnosis" not in case_study["OSCE_Examination"]
        or "体格检查" not in case_study
        or "辅助检查" not in case_study
    ):
        return None
    try:
        LLM_caller = LLM_Caller_for_One_Thread()
        record_inquiry = json.dumps(
            case_study["OSCE_Examination"]["Patient_Actor"],
            indent=2,
            ensure_ascii=False,
        )
        diagnosis = case_study["OSCE_Examination"]["Correct_Diagnosis"]
        res_dict = {"primary": case_study["辅助检查"], "extra": {}}
        for item in auxiliary_exam_list:
            full_record = (
                f"病史信息:\n{record_inquiry}\n"
                + f"部分体格检查结果:\n{case_study['体格检查']}\n"
                + f"辅助检查结果:\n{json.dumps(res_dict, ensure_ascii=False, indent=2)}\n\n"
                + diagnosis
            )
            for _ in range(try_cnt):
                try:
                    ans = LLM_caller.query_model(
                        model_str=model_name,
                        prompt=prompt_makeup_one_exam_item.format(full_record, item),
                        system_prompt=None,
                        ensure_label="answer",
                    )
                    hit_str = extract_label(ans, "hit")
                    result = extract_label(ans, "answer")
                    if result:
                        res_dict["extra"][item] = result
                        break
                except:
                    continue
        case_study["OSCE_Examination"]["Test_Results"] = res_dict
        return case_study
    except:
        return None


def judge_quality(
    case_study: dict,
    model_name: str,
    try_cnt=2,
):
    if "OSCE_Examination" not in case_study:
        return None
    case_str = json.dumps(case_study["OSCE_Examination"], indent=2, ensure_ascii=False)
    LLM_caller = LLM_Caller_for_One_Thread()
    key_name = f"quality_flag_{model_name}"
    if key_name in case_study and case_study[key_name] is not None:
        return case_study
    case_study[key_name] = None
    for _ in range(try_cnt):
        try:
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=prompt_verify_quality.format(case_str),
                system_prompt=None,
                ensure_label="answer",
            )
            if ans and ans.lower().startswith("yes"):
                case_study[key_name] = True
            else:
                case_study[key_name] = False
            return case_study
        except Exception as e:
            continue
    return case_study


def train_set_pipelines(case_study_list: list):
    func_LLM_list = [
        # 判断是否有且只有一个病人
        (judge_one_patient, "cheap_llm"),
        # 判断是否覆盖问诊+体检+辅检三个阶段的病历
        (judge_coverage_3_stages, "cheap_llm"),
        # 提取核心诊断+完整诊断
        (get_full_diagnosis, "expensive_llm"),
        # 判断原始文本是否支持诊断
        (judge_enough_data_for_diagnosis, "cheap_llm"),
        # 提取现病史等病史信息
        (extract_HPI_dict, "expensive_llm"),
        # 用病人的口吻重写主诉
        (rewrite_chief_complaint, "cheap_llm"),
        # 提取体格检查+辅助检查文本
        (extract_physical_and_auxiliary, "expensive_llm"),
        # 补全现病史
        (makeup_HPI, "expensive_llm"),
        # 编造必要的体格检查
        (makeup_necessary_physical_exam_results, "cheap_llm"),
        # 把一段辅助检查文本，dfs分类
        (classify_auxiliary_index, "cheap_llm"),
        # 判断提取的信息是否足够支持诊断
        (judge_enough_data_for_diagnosis, "cheap_llm"),
        # 判断提取质量
        (judge_quality, "cheap_llm"),
        # 难度分级
        (extract_difficulty_level, "cheap_llm"),
        # 打补丁：部分检查被分类到“其他”，需要重命名；
        (get_item_name_based_on_text, "cheap_llm"),
        # 打补丁：部分检查结果是空的，剔除
        (check_item_results, "cheap_llm"),
        # 科室分类
        (classify_research_department, "cheap_llm"),
        # 判断是否存在误诊的情况
        (judge_misdiagnosis_and_doctor, "expensive_llm"),
        # 判断病人的位置
        (judge_location, "cheap_llm"),
    ]
    # pipelines
    for idx, (func, model_type) in enumerate(func_LLM_list):
        output_temp_path = os.path.join(
            args.temp_dir, f"{idx}-after-{func.__name__}.json"
        )
        if os.path.exists(output_temp_path):
            case_study_list = read_json(output_temp_path)
            continue
        case_study_list = [x for x in case_study_list if x]
        case_study_list = batch_process_parallel(
            func=func,
            args_list=[
                [case_study, config_dict.get(model_type, model_type)]
                for case_study in case_study_list
            ],
            num_processes=100,
            use_tqdm=True,
        )
        write_json(case_study_list, output_temp_path)

    final_case_study_list = []
    # 分配id
    for idx, item in enumerate(case_study_list):
        new_item = {"OSCE_id": idx}
        new_item.update(item)
        final_case_study_list.append(new_item)
    write_json(final_case_study_list, args.dst_path)


def test_set_pipelines(case_study_list: list):
    func_LLM_list = [
        # 判断是否有且只有一个病人
        (judge_one_patient, "cheap_llm"),
        # 判断是否覆盖问诊+体检+辅检三个阶段的病历
        (judge_coverage_3_stages, "cheap_llm"),
        # 提取核心诊断+完整诊断
        (get_full_diagnosis, "expensive_llm"),
        # 判断原始文本是否支持诊断
        (judge_enough_data_for_diagnosis, "cheap_llm"),
        # 提取现病史等病史信息
        (extract_HPI_dict, "expensive_llm"),
        # 用病人的口吻重写主诉
        (rewrite_chief_complaint, "cheap_llm"),
        # 提取体格检查+辅助检查文本
        (extract_physical_and_auxiliary, "expensive_llm"),
        # 补全现病史
        (makeup_HPI, "cheap_llm"),
        # 给出全部的体格检查
        (makeup_all_physical_exam_results, "cheap_llm"),
        # 给出常见的辅助检查
        (makeup_common_auxiliary_results, "cheap_llm"),
        # 判断提取的信息是否足够支持诊断
        (judge_enough_data_for_diagnosis, "expensive_llm"),
        # 判断提取质量
        (judge_quality, "expensive_llm"),
        # 难度分级
        (extract_difficulty_level, "cheap_llm"),
        # 科室分类
        (classify_research_department, "cheap_llm"),
        # 判断是否存在误诊的情况
        (judge_misdiagnosis_and_doctor, "expensive_llm"),
        # 判断病人的位置
        (judge_location, "cheap_llm"),
        (map_physical_name, "cheap_llm"),
    ]
    # pipelines
    for idx, (func, model_type) in enumerate(func_LLM_list):
        output_temp_path = os.path.join(
            args.temp_dir, f"{idx}-after-{func.__name__}.json"
        )
        if os.path.exists(output_temp_path):
            case_study_list = read_json(output_temp_path)
            continue
        case_study_list = [x for x in case_study_list if x]
        case_study_list = batch_process_parallel(
            func=func,
            args_list=[
                [case_study, config_dict.get(model_type, model_type)]
                for case_study in case_study_list
            ],
            num_processes=100,
            use_tqdm=True,
        )
        write_json(case_study_list, output_temp_path)

    final_case_study_list = []
    # 分配id
    for idx, item in enumerate(case_study_list):
        new_item = {"OSCE_id": idx}
        new_item.update(item)
        final_case_study_list.append(new_item)
    write_json(final_case_study_list, args.dst_path)


class EvaluateConfig:
    def __init__(
        self,
        src_path: str = "input.json",
        dst_path: str = "output.json",
        temp_dir: str = "temp_json/",
        dataset_type: str = "test_set",
    ):
        self.src_path = src_path
        self.dst_path = dst_path
        self.temp_dir = temp_dir
        self.dataset_type = dataset_type

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_command_line(cls) -> "EvaluateConfig":
        default_config = cls()
        parser = ArgumentParser()
        parser.add_argument(
            "--src_path",
            type=str,
            default=default_config.src_path,
        )
        parser.add_argument(
            "--dst_path",
            type=str,
            default=default_config.dst_path,
        )
        parser.add_argument(
            "--temp_dir",
            type=str,
            default=default_config.temp_dir,
        )
        parser.add_argument(
            "--dataset_type",
            type=str,
            default=default_config.dataset_type,
        )

        args = parser.parse_args()
        return cls(**vars(args))


if __name__ == "__main__":
    args = EvaluateConfig().from_command_line()
    os.makedirs(args.temp_dir, exist_ok=True)
    raw_case_study_list = read_json(args.src_path)
    case_study_list = []
    for case_study in raw_case_study_list:
        if "raw_data" in case_study:
            case_study_list.append(case_study)
        else:
            case_study_list.append({"raw_data": case_study})
    if args.dataset_type == "test_set":
        test_set_pipelines(case_study_list)
    else:
        train_set_pipelines(case_study_list)
