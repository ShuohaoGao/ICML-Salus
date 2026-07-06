from argparse import ArgumentParser
from typing import List, Union


class EvaluateConfig:
    def __init__(
        self,
        inf_type: str = "llm",
        cheap_llm: str = "deepseek-v3-0324",
        expensive_llm: str = "deepseek-r1-0528",
        dataset_path: Union[
            str, List[str]
        ] = "agentclinic_medqa.json",  # Changed to accept both str and List[str]
        num_scenarios=None,
        inquiry_num: int = 10,
        physical_exam_num: int = 50,
        auxiliary_exam_num: int = 10,
        chinese: bool = False,
        parallel_thread_num: int = 100,
        repeat_num: int = 1,
        continue_file_path: str = None,
    ):
        self.inf_type = inf_type
        self.cheap_llm = cheap_llm
        self.expensive_llm = expensive_llm
        # Ensure dataset_path is always a list
        self.dataset_path = (
            [dataset_path] if isinstance(dataset_path, str) else dataset_path
        )
        self.num_scenarios = num_scenarios
        self.inquiry_num = inquiry_num
        self.physical_exam_num = physical_exam_num
        self.auxiliary_exam_num = auxiliary_exam_num
        self.Chinese = chinese
        self.parallel_thread_num = parallel_thread_num
        self.repeat_num = repeat_num
        self.continue_file_path = continue_file_path

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_command_line(cls) -> "EvaluateConfig":
        default_config = cls()
        parser = ArgumentParser(description="Medical Diagnosis Simulation")

        parser.add_argument(
            "--inf_type",
            type=str,
            choices=["llm", "human_doctor", "human_patient"],
            default=default_config.inf_type,
            help=f"Type of inference agent. Default: {default_config.inf_type}",
        )
        parser.add_argument(
            "--cheap_llm",
            type=str,
            default=default_config.cheap_llm,
            help=f"A cheaper, faster LLM for simple tasks. Default: {default_config.cheap_llm}",
        )
        parser.add_argument(
            "--expensive_llm",
            type=str,
            default=default_config.expensive_llm,
            help=f"A more powerful LLM for complex tasks. Default: {default_config.expensive_llm}",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            nargs="+",  # Accept one or more arguments
            default=default_config.dataset_path,
            help=f"List of dataset paths. Default: {' '.join(default_config.dataset_path)}",
        )
        parser.add_argument(
            "--num_scenarios",
            type=int,
            default=default_config.num_scenarios,
            required=False,
            help="Number of scenarios to simulate. Default: all scenarios in the dataset.",
        )
        parser.add_argument(
            "--inquiry_num",
            type=int,
            default=default_config.inquiry_num,
            required=False,
            help=f"Max number of inquiry turns. Default: {default_config.inquiry_num}",
        )
        parser.add_argument(
            "--physical_exam_num",
            type=int,
            default=default_config.physical_exam_num,
            required=False,
            help=f"Max number of physical exam turns. Default: {default_config.physical_exam_num}",
        )
        parser.add_argument(
            "--auxiliary_exam_num",
            type=int,
            default=default_config.auxiliary_exam_num,
            required=False,
            help=f"Max number of auxiliary exam turns. Default: {default_config.auxiliary_exam_num}",
        )
        # For boolean flags, 'action' is sufficient. The default is False, which matches the __init__.
        parser.add_argument(
            "--Chinese",
            dest="chinese",
            action="store_true",
            help="Use Chinese prompts. Default is False (English).",
        )
        parser.add_argument(
            "--parallel_thread_num",
            type=int,
            default=default_config.parallel_thread_num,
            required=False,
            help=f"Number of parallel threads. Default: {default_config.parallel_thread_num}",
        )
        parser.add_argument(
            "--repeat_num",
            type=int,
            default=default_config.repeat_num,
            required=False,
            help=f"Number of conversations to generate for each record. Default: {default_config.repeat_num}",
        )
        parser.add_argument(
            "--continue_file_path",
            type=str,
            default=default_config.continue_file_path,
            required=False,
            help="Path to a file to continue a previous run. Default: None",
        )

        args = parser.parse_args()
        return cls(**vars(args))


args = EvaluateConfig().from_command_line()
