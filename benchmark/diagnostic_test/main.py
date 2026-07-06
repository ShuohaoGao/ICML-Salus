import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from doctor_factory import create_doctor_agent
from environment_real import Environment
from utils.call_LLM import LLM_Caller_for_One_Thread, launch_vllm_server
from utils.data_loader import Scenario_OSCE, Scenario_OSCE_Loader
from utils.diagnosis_eval import diagnose_from_record_and_judge_correctness, judge_topk
from utils.evaluate_config import args
from utils.io_func import read_json, write_json
from utils.tools import batch_process_parallel


global_dict_from_sys_prompt_to_context_id = {}


def run_one_scenario(_scenario_id: int, scenario: Scenario_OSCE, _repeat: int = 1):
    """
    使用单个病历进行评测；本函数可以并发调用。
    """
    try:
        env = Environment(scenario_id=_scenario_id, scenario=scenario)
        LLM_caller = LLM_Caller_for_One_Thread(
            introduction_log=f"\n<hr>\n\n## {_scenario_id} Doctor\n",
            dict_from_sys_prompt_to_context_id=global_dict_from_sys_prompt_to_context_id.copy(),
        )

        doctor_agent = create_doctor_agent(
            class_name=args.doctor_class,
            LLM_caller=LLM_caller,
            model_name=args.doctor_llm,
        )

        answer_to_doctor = env.initial_answer_to_doctor()
        while True:
            question_from_doctor = doctor_agent.inference(answer_to_doctor)
            answer_to_doctor = env.response_to_doctor(question_from_doctor)
            if answer_to_doctor is None:
                break

        info_dict = env.info_dict()
        info_dict.update(
            {
                "doctor_detailed_log": LLM_caller.LLM_log_list,
                "top3_diagnosis": doctor_agent.get_top3_diagnosis(),
                "repeat_id": _repeat,
            }
        )

        return info_dict, _scenario_id, _repeat
    except Exception as e:
        print("Error:", _scenario_id, _repeat, e, flush=True)
        return None, _scenario_id, _repeat


def main():
    """
    运行诊断测试，然后保存原始结果；指标汇总由 eval() 负责。
    """
    log_path = f"detailed_log/{args.eval_id}-full-log.json"
    scenario_loader = Scenario_OSCE_Loader(args.dataset_path)
    results_list = []
    if os.path.exists(log_path):
        results_list = read_json(log_path).get("results_list", [])
        print(f"Resume from {log_path}: {len(results_list)} finished.", flush=True)
    finished = {(res["scenario_id"], res["repeat_id"]) for res in results_list}

    args.num_scenarios = min(args.num_scenarios, scenario_loader.num_scenarios)
    total_num_scenarios = args.num_scenarios
    os.makedirs("detailed_log", exist_ok=True)
    pool = (
        ThreadPoolExecutor
        if "baichuan" in args.doctor_llm.lower()
        else ProcessPoolExecutor
    )

    with pool(max_workers=args.parallel_thread_num) as executor:
        future_to_scenario = {}
        for _repeat in range(args.repeat_cnt):
            for _scenario_id in range(total_num_scenarios):
                if (_scenario_id, _repeat) in finished:
                    continue
                future = executor.submit(
                    run_one_scenario,
                    _scenario_id,
                    scenario_loader.get_scenario(_scenario_id),
                    _repeat,
                )
                future_to_scenario[future] = _scenario_id

        for future in as_completed(future_to_scenario):
            info_dict, _scenario_id, _repeat = future.result()
            if info_dict:
                print(f"Scene {_scenario_id}-{_repeat}: done", flush=True)
                results_list.append(info_dict)
                if len(results_list) % 50 == 0:
                    write_json(
                        {"settings": args.to_dict(), "results_list": results_list},
                        log_path,
                    )
                    print(f"Saved intermediate results for {len(results_list)} scenarios.", flush=True)
            else:
                print(f"Scene {_scenario_id}-{_repeat}: skipped", flush=True)

    results_list = sorted(results_list, key=lambda x: (x["scenario_id"], x["repeat_id"]))
    write_json(
        {"settings": args.to_dict(), "results_list": results_list},
        log_path,
    )


def _safe_avg(total_value, total_count):
    if total_count == 0:
        return 0.0
    return total_value / total_count


def _first_item(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _count_auxiliary_items(res: dict) -> int:
    cnt = 0
    for item in res["print_dialog"]:
        if "请求进行以下辅助检查：" not in item:
            continue
        item_lines = item.split("以下辅助检查：")[-1].split("\n")
        cnt += sum(1 for line in item_lines if len(line) > 2 and line[0] == "-")
    return cnt


def _print_summary_block(title: str, lines: list[str]):
    print(f"\n## {title}")
    for line in lines:
        print(line)


def eval():
    """
    评测并更新 detailed_log 中的 summary。
    """
    saved_info = read_json(f"detailed_log/{args.eval_id}-full-log.json")
    settings = saved_info["settings"]
    results_list = saved_info["results_list"]
    total_num_scenarios = settings["num_scenarios"] * settings["repeat_cnt"]
    scenario_loader = Scenario_OSCE_Loader(args.dataset_path)

    args.parallel_thread_num = 100

    print("\n".join([_log for res in results_list for _log in res["print_dialog"]]))

    total_correct_count_final = sum(
        bool(_first_item(res["correctness"], False)) for res in results_list
    )
    total_auxiliary_rounds = sum(
        _first_item(res["interaction_count"], [0, 0, 0])[-1]
        if isinstance(_first_item(res["interaction_count"]), list)
        else res["interaction_count"][-1]
        for res in results_list
    )
    avg_aux_hit = _safe_avg(
        sum(res.get("aux_hit", 0.0) for res in results_list),
        len(results_list),
    )
    avg_auxiliary_items = _safe_avg(
        sum(_count_auxiliary_items(res) for res in results_list),
        len(results_list),
    )

    _print_summary_block(
        "Overall",
        [
            f"Final accuracy: {_safe_avg(total_correct_count_final, total_num_scenarios) * 100:.2f}%",
            f"Avg auxiliary-stage requests: {_safe_avg(total_auxiliary_rounds, total_num_scenarios):.2f}",
            f"Avg auxiliary items: {avg_auxiliary_items:.2f}",
            f"Avg aux hit: {avg_aux_hit * 100:.2f}%",
        ],
    )

    level_dict = {level: {"correct": 0, "cases": 0, "inquiry": 0} for level in range(1, 6)}
    for res in results_list:
        level_info = level_dict[res["difficulty"]]
        level_info["correct"] += bool(_first_item(res["correctness"], False))
        level_info["cases"] += 1
        level_info["inquiry"] += sum(res["interaction_count"])

    _print_summary_block(
        "By Difficulty",
        [
            f"Level {level}: accuracy={_safe_avg(level_dict[level]['correct'], level_dict[level]['cases']) * 100:.3f}% "
            f"avg_inquiry={_safe_avg(level_dict[level]['inquiry'], level_dict[level]['cases']):.2f} "
            f"cases={level_dict[level]['cases']}"
            for level in level_dict
        ],
    )

    level_cnt = {"完全正确": 0, "临床可接受": 0, "部分正确": 0, "不正确": 0}
    for res in results_list:
        level = _first_item(res["correctness_level"], "不正确")
        level_cnt[level] = level_cnt.get(level, 0) + 1

    _print_summary_block(
        "Correctness Level",
        [
            f"{level}: {level_cnt.get(level, 0)} cases, "
            f"{_safe_avg(level_cnt.get(level, 0), total_num_scenarios) * 100:.2f}%"
            for level in ["完全正确", "临床可接受", "部分正确", "不正确"]
        ],
    )

    res_rediagnose_list = batch_process_parallel(
        func=diagnose_from_record_and_judge_correctness,
        args_list=[
            (
                res["dialog_no_diagnosis"],
                res["correct_diagnosis"],
                args.rediagnosis_llm,
                None,
                scenario_loader.get_scenario(res["scenario_id"]).full_record(),
            )
            for res in results_list
        ],
        num_processes=args.parallel_thread_num,
        use_tqdm=False,
    )
    for idx in range(len(results_list)):
        results_list[idx].update({"rediagnosis": res_rediagnose_list[idx][0]})
    rediagnosis_accuracy = _safe_avg(
        sum(item[0]["correctness"] for item in res_rediagnose_list),
        total_num_scenarios,
    )

    res_topk_list = batch_process_parallel(
        func=judge_topk,
        args_list=[
            (
                res["top3_diagnosis"],
                res["correct_diagnosis"],
                args.rediagnosis_llm,
                None,
                scenario_loader.get_scenario(res["scenario_id"]).full_record(),
            )
            for res in results_list
        ],
        num_processes=args.parallel_thread_num,
        use_tqdm=False,
    )
    for idx in range(len(results_list)):
        results_list[idx].update({"top3_diagnosis": res_topk_list[idx][0]})

    top1_correct_count = sum(res["top3_diagnosis"][0]["correctness"] for res in results_list)
    top2_correct_count = sum(
        any(res["top3_diagnosis"][idx]["correctness"] for idx in range(2))
        for res in results_list
    )
    top3_correct_count = sum(
        any(res["top3_diagnosis"][idx]["correctness"] for idx in range(3))
        for res in results_list
    )

    summary = {
        "overall": {
            "final_accuracy": _safe_avg(total_correct_count_final, total_num_scenarios),
            "avg_auxiliary_stage_requests": _safe_avg(
                total_auxiliary_rounds, total_num_scenarios
            ),
            "avg_auxiliary_items": avg_auxiliary_items,
            "avg_aux_hit": avg_aux_hit,
        },
        "by_difficulty": {
            str(level): {
                "accuracy": _safe_avg(
                    level_dict[level]["correct"], level_dict[level]["cases"]
                ),
                "avg_inquiry": _safe_avg(
                    level_dict[level]["inquiry"], level_dict[level]["cases"]
                ),
                "cases": level_dict[level]["cases"],
            }
            for level in level_dict
        },
        "correctness_level": {
            level: {
                "cases": level_cnt.get(level, 0),
                "ratio": _safe_avg(level_cnt.get(level, 0), total_num_scenarios),
            }
            for level in ["完全正确", "临床可接受", "部分正确", "不正确"]
        },
        "rediagnosis": {
            "model": args.rediagnosis_llm,
            "accuracy": rediagnosis_accuracy,
        },
        "topk": {
            "top1_accuracy": _safe_avg(top1_correct_count, total_num_scenarios),
            "top2_accuracy": _safe_avg(top2_correct_count, total_num_scenarios),
            "top3_accuracy": _safe_avg(top3_correct_count, total_num_scenarios),
        },
        "eval_LLM_log_list": [
            log_item
            for result in res_rediagnose_list + res_topk_list
            for log_item in result[1]
        ],
    }

    _print_summary_block(
        "Re-diagnosis",
        [
            f"Model: {args.rediagnosis_llm}",
            f"Accuracy: {rediagnosis_accuracy * 100:.2f}%",
        ],
    )
    _print_summary_block(
        "Top-k",
        [
            f"Top-1 Accuracy: {_safe_avg(top1_correct_count, total_num_scenarios) * 100:.2f}%",
            f"Top-2 Accuracy: {_safe_avg(top2_correct_count, total_num_scenarios) * 100:.2f}%",
            f"Top-3 Accuracy: {_safe_avg(top3_correct_count, total_num_scenarios) * 100:.2f}%",
        ],
    )

    write_json(
        {
            "settings": args.to_dict(),
            "results_list": results_list,
            "summary": summary,
        },
        f"detailed_log/{args.eval_id}-full-log.json",
    )


if __name__ == "__main__":
    if not args.eval_id:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        args.eval_id = (
            f"{current_time}-{Path(args.dataset_path).stem}"
            f"--{args.doctor_class}={args.doctor_llm.split('/')[-1]}"
        )
        print(args.eval_id, flush=True)
    else:
        print(f"Using eval_id: {args.eval_id}", flush=True)

    if args.doctor_llm_path:
        launch_vllm_server(
            model_path=args.doctor_llm_path, model_name=args.doctor_llm
        )
    try:
        main()
    except Exception as e:
        print("Error in main:", e, flush=True)

    if args.evaluate:
        eval()
