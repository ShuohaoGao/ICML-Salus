import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from doctor_factory import create_doctor_agent
from utils.call_LLM import LLM_Caller_for_One_Thread, launch_vllm_server
from utils.data_loader import Scenario_OSCE, Scenario_OSCE_Loader
from utils.diagnosis_eval import judge_topk
from utils.evaluate_config import args
from utils.io_func import read_json, write_json




def summarize_record(scenario: Scenario_OSCE) -> str:
    return scenario.full_record()


def _log_path() -> str:
    return f"detailed_log_static/{args.eval_id}-full-log.json"


def run_one_scenario(_scenario_id: int, scenario: Scenario_OSCE, _repeat: int = 1):
    """
    使用单个病历进行静态诊断评测；本函数可以并发调用。
    """
    try:
        LLM_caller = LLM_Caller_for_One_Thread(
            introduction_log=f"\n<hr>\n\n## {_scenario_id} Doctor\n",
        )

        doctor_agent = create_doctor_agent(
            class_name=args.doctor_class,
            LLM_caller=LLM_caller,
            model_name=args.doctor_llm,
        )

        record_summary = summarize_record(scenario)
        static_top3_diagnosis_list = doctor_agent.get_top3_diagnosis(record_summary)
        correctness_list, _ = judge_topk(
            topk_diagnosis=static_top3_diagnosis_list,
            correct_diagnosis=scenario.diagnosis_information(),
            model_name=args.judge_correctness_llm,
            LLM_caller=LLM_caller,
            other_info=scenario.full_record(),
        )
        dialog_list = [
            f"\n### {_scenario_id}\n",
            f"{record_summary}\n",
            f"Correct Diagnosis: {scenario.diagnosis_information()}\n",
        ]
        for idx, item in enumerate(correctness_list):
            dialog_list.append(
                f"Diagnosis {idx + 1}: {item['diagnosis']}\n"
                + ("**CORRECT**" if item["correctness"] else "**INCORRECT**")
                + f" {item['level']}\n"
            )
        info_dict = {
            "scenario_id": _scenario_id,
            "repeat_id": _repeat,
            "correct_diagnosis": scenario.diagnosis_information(),
            "difficulty": scenario.difficulty_level(),
            "print_dialog": dialog_list,
            "top3_diagnosis": correctness_list,
            "detailed_log": LLM_caller.LLM_log_list,
        }
        return info_dict, _scenario_id, _repeat
    except Exception as e:
        print("Error:", _scenario_id, _repeat, e, flush=True)
        return None, _scenario_id, _repeat


def main():
    """
    运行静态诊断测试，然后保存原始结果；支持断点续跑和中间保存。
    """
    log_path = _log_path()
    scenario_loader = Scenario_OSCE_Loader(args.dataset_path)
    results_list = []

    if os.path.exists(log_path):
        results_list = read_json(log_path).get("results_list", [])
        print(f"Resume from {log_path}: {len(results_list)} finished.", flush=True)
    finished = {
        (res["scenario_id"], res.get("repeat_id", 0))
        for res in results_list
    }

    args.num_scenarios = min(args.num_scenarios, scenario_loader.num_scenarios)
    total_num_scenarios = args.num_scenarios
    os.makedirs("detailed_log_static", exist_ok=True)
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
                    print(
                        f"Saved intermediate results for {len(results_list)} scenarios.",
                        flush=True,
                    )
            else:
                print(f"Scene {_scenario_id}-{_repeat}: skipped", flush=True)

    results_list = sorted(
        results_list,
        key=lambda x: (x["scenario_id"], x.get("repeat_id", 0)),
    )
    write_json(
        {"settings": args.to_dict(), "results_list": results_list},
        log_path,
    )


def _safe_avg(total_value, total_count):
    if total_count == 0:
        return 0.0
    return total_value / total_count


def eval():
    """
    评测并更新 detailed_log_static 中的 summary。
    """
    log_path = _log_path()
    saved_info = read_json(log_path)
    settings = saved_info["settings"]
    results_list = saved_info["results_list"]
    total_num_scenarios = settings["num_scenarios"] * settings["repeat_cnt"]

    print("\n".join([_log for res in results_list for _log in res["print_dialog"]]))

    level_cnt = {"完全正确": 0, "临床可接受": 0, "部分正确": 0, "不正确": 0}
    for res in results_list:
        level = res["top3_diagnosis"][0]["level"]
        level_cnt[level] = level_cnt.get(level, 0) + 1
    for level in ["完全正确", "临床可接受", "部分正确", "不正确"]:
        print(
            f"{level}: {level_cnt.get(level, 0)} cases, "
            f"{_safe_avg(level_cnt.get(level, 0), total_num_scenarios) * 100:.2f}%"
        )

    top1_correct_count = sum(
        res["top3_diagnosis"][0]["correctness"] for res in results_list
    )
    top2_correct_count = sum(
        any(
            [
                res["top3_diagnosis"][0]["correctness"],
                res["top3_diagnosis"][1]["correctness"],
            ]
        )
        for res in results_list
    )
    top3_correct_count = sum(
        any(
            [
                res["top3_diagnosis"][0]["correctness"],
                res["top3_diagnosis"][1]["correctness"],
                res["top3_diagnosis"][2]["correctness"],
            ]
        )
        for res in results_list
    )
    print(
        f"Top-1 Accuracy: {_safe_avg(top1_correct_count, total_num_scenarios) * 100:.2f}%",
        f"\nTop-2 Accuracy: {_safe_avg(top2_correct_count, total_num_scenarios) * 100:.2f}%",
        f"\nTop-3 Accuracy: {_safe_avg(top3_correct_count, total_num_scenarios) * 100:.2f}%",
    )

    summary = {
        "correctness_level": {
            level: {
                "cases": level_cnt.get(level, 0),
                "ratio": _safe_avg(level_cnt.get(level, 0), total_num_scenarios),
            }
            for level in ["完全正确", "临床可接受", "部分正确", "不正确"]
        },
        "topk": {
            "top1_accuracy": _safe_avg(top1_correct_count, total_num_scenarios),
            "top2_accuracy": _safe_avg(top2_correct_count, total_num_scenarios),
            "top3_accuracy": _safe_avg(top3_correct_count, total_num_scenarios),
        },
    }

    write_json(
        {
            "settings": args.to_dict(),
            "results_list": results_list,
            "summary": summary,
        },
        log_path,
    )


if __name__ == "__main__":
    if not args.eval_id:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        args.eval_id = (
            f"static-{current_time}-{Path(args.dataset_path).stem}"
            f"--doctor={args.doctor_llm.split('/')[-1]}"
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
