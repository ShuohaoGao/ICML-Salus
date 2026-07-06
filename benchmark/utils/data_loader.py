import json
import random
from typing import Union, List


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class Scenario_OSCE:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.chief_complaint = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"][
            "Physical_Examination_Findings"
        ]
        self.core_diagnosis = scenario_dict["OSCE_Examination"].get(
            "core_diagnosis", ""
        )
        self.full_diagnosis = scenario_dict["OSCE_Examination"].get(
            "full_diagnosis", ""
        )
        if isinstance(self.diagnosis, list) or isinstance(self.diagnosis, dict):
            self.diagnosis = json.dumps(self.diagnosis, ensure_ascii=False)

        if self.diagnosis.startswith('"'):
            self.diagnosis = self.diagnosis[1:]
        if self.diagnosis.endswith('"'):
            self.diagnosis = self.diagnosis[:-1]

        if self.chief_complaint.startswith('"'):
            self.chief_complaint = self.chief_complaint[1:]
        if self.chief_complaint.endswith('"'):
            self.chief_complaint = self.chief_complaint[:-1]

    def patient_information(self) -> str:
        return json.dumps(self.patient_info, ensure_ascii=False)

    def chief_complaint_info(self) -> str:
        return self.chief_complaint

    def physical_exam_information(self) -> str:
        return json.dumps(self.physical_exams, ensure_ascii=False)

    def short_physical_exam_information(self) -> str:
        if "体格检查" in self.scenario_dict:
            return f"<体格检查>\n{self.scenario_dict['体格检查']}\n</体格检查>"
        return json.dumps(self.physical_exams, ensure_ascii=False)

    def test_information(self) -> str:
        return json.dumps(self.tests, ensure_ascii=False)

    def diagnosis_information(self) -> str:
        return self.diagnosis

    def difficulty_level(self) -> int:
        for i in range(1, 6):
            if str(i) in self.scenario_dict["difficulty"]:
                return i
        return None

    def cover_3_stages(self) -> bool:
        for d in [
            self.patient_information(),
            self.physical_exam_information(),
            self.test_information(),
        ]:
            # 过滤掉太短的病历
            if len(d) < 10:
                return False
        return True

    def full_record(self) -> str:
        res = "<病史信息>\n" + self.patient_information() + "\n</病史信息>\n\n"
        if "体格检查" in self.scenario_dict:
            res += "<体格检查>\n" + str(self.scenario_dict["体格检查"]) + "\n</体格检查>\n\n"
        else:
            res += "体格检查结果：\n" + self.physical_exam_information() + "\n\n"
        if "辅助检查" in self.scenario_dict:
            res += "<辅助检查>\n" + str(self.scenario_dict["辅助检查"]) + "\n</辅助检查>\n"
        else:
            res += "<辅助检查>\n" + self.test_information() + "\n</辅助检查>\n"
        return res

    def core_diagnosis_info(self) -> str:
        return self.core_diagnosis

    def full_diagnosis_info(self) -> str:
        return self.full_diagnosis

    def record_for_patient_agent(self) -> str:
        res = "<病史信息>\n" + self.patient_information() + "\n</病史信息>\n\n"
        res += self.diagnosis_information()
        return res

    def visible_record_of_patient_agent(self) -> str:
        return self.patient_information()

    def invisible_record_of_patient_agent(self) -> str:
        physical_exam_info = "体格检查结果:\n"
        if "体格检查" in self.scenario_dict:
            physical_exam_info += self.scenario_dict["体格检查"]
        else:
            physical_exam_info += self.physical_exam_information()
        auxiliary_exam_info = "辅助检查结果:\n"
        if "辅助检查" in self.scenario_dict:
            auxiliary_exam_info += str(self.scenario_dict["辅助检查"])
        else:
            auxiliary_exam_info += self.test_information()

        record = physical_exam_info + "\n" + auxiliary_exam_info + "\n\n"
        record += self.diagnosis_information()
        return record

    def record_for_physical_exam_agent(self) -> str:
        res = "<病史信息>\n" + self.patient_information() + "\n</病史信息>\n\n"
        if "体格检查" in self.scenario_dict:
            res += "<体格检查>\n" + self.scenario_dict["体格检查"] + "\n</体格检查>\n\n"
        else:
            res += "体格检查结果：\n" + self.physical_exam_information() + "\n\n"
        if "辅助检查" in self.scenario_dict:
            res += "<辅助检查>\n" + str(self.scenario_dict["辅助检查"]) + "\n</辅助检查>\n"
        else:
            res += "<辅助检查>\n" + self.test_information() + "\n</辅助检查>\n"
        res += self.diagnosis_information()
        return res

    def record_for_auxiliary_exam_agent(self) -> str:
        res = "<病史信息>\n" + self.patient_information() + "\n</病史信息>\n\n"
        if "体格检查" in self.scenario_dict:
            res += "<体格检查>\n" + self.scenario_dict["体格检查"] + "\n</体格检查>\n\n"
        else:
            res += "体格检查结果：\n" + self.physical_exam_information() + "\n\n"
        if "辅助检查" in self.scenario_dict:
            res += "<辅助检查>\n" + str(self.scenario_dict["辅助检查"]) + "\n</辅助检查>\n"
        else:
            res += "<辅助检查>\n" + self.test_information() + "\n</辅助检查>\n"
        res += self.diagnosis_information()
        return res


class Scenario_OSCE_Loader:
    """
    支持合并多个json病历
    """

    def __init__(
        self, file_path: Union[str, List[str]] = "agentclinic_medqa.json"
    ) -> None:
        paths = [file_path] if isinstance(file_path, str) else file_path
        raw_scenario_list = []
        for path in paths:
            file_data = read_json(path)
            raw_scenario_list.extend(file_data)

        self.scenarios = [Scenario_OSCE(scenario) for scenario in raw_scenario_list]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]
