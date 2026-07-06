import sys

sys.path.append("../")

from utils.call_LLM import LLM_Caller_for_One_Thread
from utils.data_loader import Scenario_OSCE
from utils.diagnosis_eval import judge_correct_diagnosis_level
from utils.evaluate_config import args
from utils.medical_exam import MedTestAgent, Warning_from_too_many_auxiliary_exam


class Environment:
    """
    run one diagnostic test env

    The doctor receives patient history and physical examination findings at the
    beginning, then can only request auxiliary exams before making the final
    diagnosis.
    """

    def __init__(self, scenario_id: int, scenario: Scenario_OSCE) -> None:
        self.scenario_id = scenario_id
        self.scenario = scenario
        self.LLM_caller = LLM_Caller_for_One_Thread(
            introduction_log=f"\n<hr>\n\n## {scenario_id} Environment\n"
        )
        self.current_stage = "Auxiliary Exam"
        self.med_test_agent = MedTestAgent(
            LLM_caller=self.LLM_caller,
            scenario=scenario,
            model_name=args.measurement_llm,
        )

        # for log
        self.dialog_list = []
        self.dialog_no_diagnosis = ""

        # for interaction tracking
        self.interaction_count = 0
        self.stage_auxiliary_exam_interaction_count = 0
        self.final_diagnosis = ""
        self.correct_diagnosis = scenario.diagnosis_information()
        self.difficulty_level = scenario.difficulty_level()
        self.aux_hit_cnt = 0
        self.aux_hit_total_cnt = 0

    def initial_answer_to_doctor(self) -> str:
        """获取初始病历信息"""
        record = "病史信息：\n" + self.scenario.patient_information() + "\n\n"
        record += (
            "体格检查结果：\n"
            + self.scenario.short_physical_exam_information()
            + "\n\n"
        )
        self.dialog_list.append(record)
        return record

    def judge_stage(self, question: str) -> None:
        """
        judge the stage of doctor:
        1. Auxiliary Exam
        2. Final Diagnosis
        """
        if any(
            word in question.lower()
            for word in ["final diagnosis", "诊断结果", "核心诊断", "完整诊断"]
        ):
            self.current_stage = "Final Diagnosis"
        else:
            self.current_stage = "Auxiliary Exam"

    def response_to_doctor(self, question_from_doctor: str) -> str:
        """Process auxiliary exam requests until the doctor gives a diagnosis."""
        if self.current_stage == "Final Diagnosis":
            raise ValueError("The dialog is already terminated!")

        self.dialog_list.append(f"Doctor: {question_from_doctor}")
        self.judge_stage(question_from_doctor)

        if self.current_stage == "Auxiliary Exam":
            self.interaction_count += 1
            self.stage_auxiliary_exam_interaction_count += 1
            if self.stage_auxiliary_exam_interaction_count >= args.auxiliary_exam_num:
                if (
                    self.stage_auxiliary_exam_interaction_count
                    >= args.auxiliary_exam_num + 2
                ):
                    raise ValueError("太多辅助检查了")
                self.answer_to_doctor = Warning_from_too_many_auxiliary_exam
                self.dialog_list.append(f"{self.answer_to_doctor}\n\n")
                return self.answer_to_doctor

            self.answer_to_doctor = self.med_test_agent.inference(question_from_doctor)
            self.aux_hit_cnt += self.med_test_agent.last_hit_cnt
            self.aux_hit_total_cnt += self.med_test_agent.last_hit_judged_cnt
            self.dialog_list.append(f"Results: {self.answer_to_doctor}")
            return self.answer_to_doctor

        assert self.current_stage == "Final Diagnosis"
        self.final_diagnosis = question_from_doctor
        self.dialog_no_diagnosis = "\n".join(self.dialog_list[:-1])
        return None

    def judge_diagnosis_correctness_and_level(
        self, diagnosis_to_be_judged: str
    ) -> tuple[bool, str]:
        return judge_correct_diagnosis_level(
            correct_diagnosis=self.scenario.diagnosis_information(),
            diagnosis_to_be_judged=diagnosis_to_be_judged,
            LLM_caller=self.LLM_caller,
            other_info=self.scenario.full_record(),
            model_name=args.judge_correctness_llm,
        )

    def info_dict(self) -> dict:
        correctness, correctness_level = self.judge_diagnosis_correctness_and_level(
            self.final_diagnosis
        )

        self.dialog_list = [f"### {self.scenario_id}"] + self.dialog_list
        self.dialog_list.append(
            f"\nScene {self.scenario_id}, Correct answer: **{self.correct_diagnosis}**"
            + f"difficulty: {self.difficulty_level}"
        )
        self.dialog_list.append(
            f"0. Final Diagnosis was \n```\n{self.final_diagnosis}\n```\n"
            + ("**CORRECT**" if correctness else "**INCORRECT**")
            + f"  {correctness_level}\n\n"
        )

        aux_hit = (
            self.aux_hit_cnt / self.aux_hit_total_cnt
            if self.aux_hit_total_cnt > 0
            else 0.0
        )
        return {
            "scenario_id": self.scenario_id,
            "correct_diagnosis": self.correct_diagnosis,
            "diagnosis": [self.final_diagnosis],
            "correctness": [correctness],
            "correctness_level": [correctness_level],
            "difficulty": self.difficulty_level,
            "interaction_count": [
                0,
                0,
                self.stage_auxiliary_exam_interaction_count,
            ],
            "print_dialog": self.dialog_list,
            "env_detailed_log": self.LLM_caller.LLM_log_list,
            "aux_hit": aux_hit,
            "dialog_no_diagnosis": self.dialog_no_diagnosis,
        }
