"""
限制一轮最多5个检查
"""

import json
import re

from .interface import BaseDoctorAgent, LLM_Caller_for_One_Thread


# 诊断Agent：只生成结构化鉴别诊断名称和概率
SYSTEM_PROMPT_LONG_DIAGNOSIS = """你是一名临床诊断专家。你的任务是基于目前已经获得的病史、体格检查和辅助检查结果，动态更新 top-5 鉴别诊断。

**核心要求：**
- 鉴别诊断必须按当前概率从高到低排序，最多 5 个；证据不足时可以少于 5 个。
- 每个诊断必须具体、完整，可以包含关键并发症或合并症，但不要把同一疾病过程的症状、并发症拆成互相竞争的多个诊断。
- 每个诊断的 `概率等级` 只能是 `高`、`中`、`低` 之一。高概率对于70%以上概率的诊断，中概率对于30-70%概率的诊断，低概率对于30%以下概率的诊断。
- 严禁给出治疗、用药或健康指导。

输出格式：
逐步的分析...
<answer>
{
  "鉴别诊断": [
    {
      "诊断名称": "具体完整的诊断1",
      "概率等级": "高"
    },
    {
      "诊断名称": "具体完整的诊断2",
      "概率等级": "中"
    }
  ]
}
</answer>

请先给出简短分析，然后在 <answer> 标签内输出严格 JSON。JSON 不要使用 Markdown 代码块，不要输出注释。
"""

USER_PROMPT_LONG_DIAGNOSIS = """以下是病人的信息：
<病历>
{}
</病历>

请基于以上病历更新结构化鉴别诊断。
"""


class DoctorAgent_Diagnosis_Verify_Decision_Action(BaseDoctorAgent):
    """
    结构化鉴别诊断。
    """

    # 初始化医生代理的对话状态和模型配置。
    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        model_name="gpt4",
    ) -> None:
        super().__init__(LLM_caller, model_name)
        self.current_summary = ""
        self.diagnosis_str = ""

    # 静态诊断目录只支持一次性根据完整病历生成 top-3 诊断。
    def get_top3_diagnosis(self, record_summary: str) -> list:
        self.current_summary = record_summary

        self.diagnosis_str = self._query(
            USER_PROMPT_LONG_DIAGNOSIS.format(self.current_summary),
            role="Doctor: differential diagnosis agent",
            system_prompt=SYSTEM_PROMPT_LONG_DIAGNOSIS,
        )
        self.top3_diagnosis = self._top_diagnosis_names(self.diagnosis_str, top_k=3)
        return self.top3_diagnosis

    # 调用 LLM 并清理 answer 标签中的输出。
    def _query(self, prompt: str, role: str, system_prompt: str) -> str:
        answer = self.LLM_caller.query_model_and_extract_label(
            model_str=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            role=role,
            ensure_label="answer",
        )
        answer = (answer or "").strip()
        answer = re.sub(r"^```(?:\w+)?\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer)
        return answer.strip()

    # 从字符串中解析 JSON 对象，兼容前后有分析文本的情况。
    def _json_object(self, value: str):
        value = (value or "").strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    # 提取排序靠前的诊断名称，供最终诊断兜底使用。
    def _top_diagnosis_names(self, diagnosis_str: str, top_k: int) -> list:
        names = []
        diagnosis_json = self._json_object(diagnosis_str)

        if isinstance(diagnosis_json, dict):
            for item in diagnosis_json.get("鉴别诊断", []):
                name = item.get("诊断名称") if isinstance(item, dict) else str(item)
                if name:
                    names.append(str(name).strip())

        if not names:
            for line in diagnosis_str.strip().splitlines():
                line = re.sub(r"^[-*]\s*|^\d+[\.\)、)\s]+", "", line.strip())
                if line and not line.startswith(("{", "}")):
                    names.append(line)

        return (names + [None] * top_k)[:top_k]
