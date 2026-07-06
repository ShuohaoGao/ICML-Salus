from .interface import BaseDoctorAgent, LLM_Caller_for_One_Thread


prompt_diagnosis_from_record = """你是一名医学专家，你的任务是根据提供的患者病历，推理出top-5鉴别诊断列表。

**核心要求：**
- **诊断排序**：诊断列表按可能性由高到低排列。
- **诊断完整性**：每个诊断都应是完整的，应具体明确（例如：使用“右下叶肺炎”而非“肺炎”；“冠心病不稳定型心绞痛”而非“心脏病”）；可包含主要疾病和相关的并发症/合并症（例如：2型糖尿病 合并 社区获得性肺炎）。
- **诊断竞争性**：列表中的各项诊断应该是相互竞争的备选方案（即鉴别诊断）。**不要将一个统一病理过程的不同方面拆分成独立的条目**（如将“社区获得性肺炎”和“发热”分别列为两个诊断）。
- **聚焦诊断**：你的回答应专注于诊断推理过程和最终的诊断列表。**严禁**提供任何治疗方案、用药建议或健康指导，也不要包含病人的检查结果等信息。
- **诊断个数**：允许鉴别诊断个数不足5个。


以下是病人的信息：
<病历>
{}
</病历>


输出格式：
逐步的分析...
<answer>
诊断1
诊断2
...
</answer>

输出示例：
...（逐步的分析）
<answer>
结核性脑膜炎/脑膜脑炎，伴有社区获得性肺炎
鼻窦旁脓肿，并发结核性全身感染
</answer>


现在请先给出逐步的分析，然后输出若干个相互竞争、完整的诊断方案，不要给出其他无内容。
"""


class Doctor_API(BaseDoctorAgent):
    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        model_name="gpt4",
    ) -> None:
        super().__init__(LLM_caller, model_name)

    def get_top3_diagnosis(self, record_summary: str) -> list:
        diag_list_str = self.LLM_caller.query_model_and_extract_label(
            model_str=self.model_name,
            prompt=prompt_diagnosis_from_record.format(record_summary),
            system_prompt=None,
            role="Doctor",
            ensure_label="answer",
            try_cnt=10,
        )
        diag_lines = diag_list_str.splitlines()
        top3 = []
        for line in diag_lines:
            line = line.strip()
            if line:
                if line.startswith("-"):
                    diag = line[1:].strip()
                elif line[0].isdigit() and "." in line:
                    diag = line.split(".", 1)[1].strip()
                else:
                    diag = line
                top3.append(diag)
            if len(top3) >= 3:
                break
        self.top3_diagnosis = (top3 + [None] * 3)[:3]
        return self.top3_diagnosis
