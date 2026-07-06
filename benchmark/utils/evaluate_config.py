from argparse import ArgumentParser


class EvaluateConfig:
    def __init__(
        self,
        doctor_class: str = "Doctor_5Agent",
        doctor_llm: str = "deepseek-v3-1-terminus",
        doctor_llm_path: str = None,
        patient_llm: str = "deepseek-v3-1-terminus",
        measurement_llm: str = "deepseek-v3-1-terminus",
        rediagnosis_llm: str = "deepseek-v3-1-terminus",
        judge_correctness_llm: str = "deepseek-v3-1-terminus",
        dataset_path: str = "/data/InteractiveMedLLM/data/OSCE_Chinese_index/0104/test/NEJM_test.json",
        num_scenarios: int = 100,
        inquiry_num: int = 50,
        physical_exam_num: int = 30,
        auxiliary_exam_num: int = 10,
        Chinese: bool = False,
        parallel_thread_num: int = 100,
        repeat_cnt: int = 3,
        evaluate: bool = False,
        eval_id: str = "",
    ):
        self.doctor_class = doctor_class
        self.doctor_llm = doctor_llm
        self.doctor_llm_path = doctor_llm_path
        self.patient_llm = patient_llm
        self.measurement_llm = measurement_llm
        self.rediagnosis_llm = rediagnosis_llm
        self.judge_correctness_llm = judge_correctness_llm
        self.dataset_path = dataset_path
        self.num_scenarios = num_scenarios
        self.inquiry_num = inquiry_num
        self.physical_exam_num = physical_exam_num
        self.auxiliary_exam_num = auxiliary_exam_num
        self.Chinese = Chinese
        self.parallel_thread_num = parallel_thread_num
        self.repeat_cnt = repeat_cnt
        self.evaluate = evaluate
        self.eval_id = eval_id

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_command_line(cls) -> "EvaluateConfig":
        parser = ArgumentParser(description="Medical Diagnosis Simulation")

        parser.add_argument(
            "--doctor_class",
            type=str,
            default=cls().doctor_class,
            help="Implement of Doctor agent",
        )
        parser.add_argument(
            "--doctor_llm",
            type=str,
            default=cls().doctor_llm,
            help="This LLM must be accessed by online LLM API",
        )
        parser.add_argument(
            "--doctor_llm_path",
            type=str,
            default=cls().doctor_llm_path,
            help="Name of the doctor (if applicable)",
        )
        parser.add_argument(
            "--patient_llm",
            type=str,
            default=cls().patient_llm,
            help="Patient Simulator",
        )
        parser.add_argument(
            "--measurement_llm",
            type=str,
            default=cls().measurement_llm,
            help="Patient Simulator that provide examination results",
        )
        parser.add_argument(
            "--rediagnosis_llm",
            type=str,
            default=cls().rediagnosis_llm,
            help="Another doctor that make a diagnosis based on the collected conversation",
        )
        parser.add_argument(
            "--judge_correctness_llm",
            type=str,
            default=cls().judge_correctness_llm,
            help="Judge whether the diagnosis is correct",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            default=cls().dataset_path,
            help="Patient records",
        )
        parser.add_argument(
            "--num_scenarios",
            type=int,
            default=cls().num_scenarios,
            help="Max number of scenarios (#patient-records) to simulate",
        )
        parser.add_argument(
            "--inquiry_num",
            type=int,
            default=cls().inquiry_num,
            help="Max number of inquiries between doctor and patient",
        )
        parser.add_argument(
            "--physical_exam_num",
            type=int,
            default=cls().physical_exam_num,
            help="Max number of physical exams",
        )
        parser.add_argument(
            "--auxiliary_exam_num",
            type=int,
            default=cls().auxiliary_exam_num,
            help="Max number of auxiliary exams",
        )
        parser.add_argument("--Chinese", action="store_true", default=cls().Chinese)
        parser.add_argument(
            "--parallel_thread_num",
            type=int,
            default=cls().parallel_thread_num,
            help="Number of parallel threads for evaluation",
        )
        parser.add_argument(
            "--repeat_cnt",
            type=int,
            default=cls().repeat_cnt,
            help="Number of times to repeat the diagnosis for each scenario to reduce randomness",
        )
        parser.add_argument("--evaluate", action="store_true", default=cls().evaluate)
        parser.add_argument(
            "--eval_id",
            type=str,
            default=cls().eval_id,
            help="Skip main() and only eval(): compute based on existing results",
        )

        args = parser.parse_args()
        return cls(**vars(args))


args = EvaluateConfig().from_command_line()
