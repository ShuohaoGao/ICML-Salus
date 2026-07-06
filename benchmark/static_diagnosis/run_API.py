import subprocess
import os


# 定义数据集列表 (List of Dictionaries)
datasets = [
    {
        "name": "NEJM",
        "path": "/m2/gsh/data/records/test/NEJM_test.json",
    },
    {
        "name": "JAMA",
        "path": "/m2/gsh/data/records/test/JAMA.json",
    },
    {
        "name": "JTO",
        "path": "/m2/gsh/data/records/test/JournalofThoracicOncology.json",
    },
    # {
    #     "name": "iiy",
    #     "path": "/m2/gsh/data/records/test/iiy.json",
    # },
]


# 定义 Doctor 列表
doctors = [
    "deepseek-v4-flash",
]

# 其他固定参数
config = {
    "class_doctor": "Doctor_API",
    # "class_doctor": "DoctorAgent_Diagnosis_Verify_Decision_Action", # 实现医生agent的class
    "patient": "deepseek-v4-flash",  # 病人扮演
    "scenario_num": "10000",  # 即全部病历
    "parallel": "30",  # 并行测试，取决于doctor LLM的并发限制；不用考虑病人扮演的并发，一般达不到上限
    "cnt_repeat": "1",  # 重复次数
    "date": "0424",  # version，不用改
}


def run_task():
    for ds in datasets:
        d_name = ds["name"]
        d_path = ds["path"]

        for doctor in doctors:
            # 构建输出文件名
            output_file = f"{d_name}-{config['date']}-{config['class_doctor']}-{doctor.split("/")[-1]}-{config['patient']}扮演-try1.md"

            print(f"\n{'='*60}")
            print(f"🚀 正在启动任务:")
            print(f"   数据集: {d_name}")
            print(f"   Doctor: {doctor}")
            print(f"   日志将保存至: {output_file}")
            print(f"{'='*60}\n")

            # 构建命令行参数列表
            cmd = [
                "python",
                "main.py",
                "--dataset_path",
                d_path,
                "--doctor_class",
                config["class_doctor"],
                "--doctor_llm",
                doctor,
                "--patient_llm",
                config["patient"],
                "--measurement_llm",
                config["patient"],
                "--rediagnosis_llm",
                config["patient"],
                "--judge_correctness_llm",
                config["patient"],
                "--num_scenarios",
                config["scenario_num"],
                "--repeat_cnt",
                config["cnt_repeat"],
                "--parallel_thread_num",
                config["parallel"],
                "--Chinese",
                "--evaluate",
            ]

            print("- ", " ".join(cmd), flush=True)

            with open(output_file, "a", encoding="utf-8") as f:
                # 启动子进程
                # stdout=f, stderr=subprocess.STDOUT 确保标准输出和报错都进同一个文件
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, text=True
                )

            if result.returncode == 0:
                print(f"✅ 任务成功完成: {d_name} + {doctor}")
            else:
                print(
                    f"❌ 任务失败 (Exit Code {result.returncode}): {d_name} + {doctor}"
                )


if __name__ == "__main__":
    run_task()
