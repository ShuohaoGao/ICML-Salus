"""
限制一轮最多5个检查
"""

import json
import re

from environment_real import (
    Warning_from_too_many_auxiliary_exam,
)
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


# 验证Agent：对单个鉴别诊断补充对应证据
SYSTEM_PROMPT_DIAGNOSIS_VERIFIER = """你是一名临床诊断证据审核专家。你的任务是基于现有病历，对输入的单个鉴别诊断补充对应证据。

**核心要求：**
- 只审核输入的这一个鉴别诊断，不要新增其他诊断。
- 根据现有病历判断该诊断的 `概率等级`，只能是 `高`、`中`、`低` 之一。高概率对于70%以上概率的诊断，中概率对于30-70%概率的诊断，低概率对于30%以下概率的诊断。当诊断缺少足够证据时，不要给出高概率；有明显矛盾时，只能是低概率。
- `支持证据` 是字符串，只写已经在病历/对话中出现的信息；多条证据用分号连接。
- `反对/矛盾证据` 是字符串，写不支持该诊断或难以解释的已知信息；如果暂时没有，写“暂无明确反对证据”。
- `待补充证据` 是字符串，写为了确认或排除该诊断需要补充的问诊、体征或检查证据；多条证据用分号连接。
- 保留输入诊断的 `诊断名称`，不要自行改名。
- 严禁给出治疗、用药或健康指导。

输出格式：
逐步的分析...
<answer>
{
  "诊断名称": "输入的具体完整诊断",
  "概率等级": "高",
  "支持证据": "证据1；证据2",
  "反对/矛盾证据": "矛盾或反对证据1",
  "待补充证据": "证据1；证据2"
}
</answer>

请详细对比诊断和病历中的每一处细节，然后在 <answer> 标签内输出严格 JSON。JSON 不要使用 Markdown 代码块，不要输出注释。
"""

USER_PROMPT_DIAGNOSIS_VERIFIER = """患者现有的病历信息：
<病历>
{}
</病历>

需要验证的鉴别诊断：
<诊断>
{}
</诊断>

请基于以上病历，为该鉴别诊断补充对应证据。
"""

# 决策动作Agent：判断是最终诊断，还是选择下一步问诊/体检/辅助检查
SYSTEM_PROMPT_DECISION_ACTION_AGENT = """你是一名医学专家，负责根据 患者现有的病历信息 和 初步鉴别诊断列表，为临床医生建议下一步是否进行辅助检查，还是直接给出最终诊断。


直接出具最终诊断的条件（满足其一即可）：
   **唯一诊断且证据确凿**: 鉴别诊断列表中仅有一个疾病，该疾病能完全解释患者的核心症状，并且已有足够的证据支持（例如“金标准”检查结果）
   **诊断层级、因果清晰**: 列表中的一个诊断是根本病因（首要诊断），而其他诊断均可被明确解释为该诊断的**直接诱因、并发症或附属表现**，而非独立的竞争性诊断。同时，首要诊断已有足够的证据支持
   **证据优势原则**: 某一个诊断的证据链条完整、逻辑严密，与病历中的所有信息（包括症状、体征、已有检查）完美契合、无任何矛盾，并且已有足够证据支持；而列表中的其他诊断均存在无法解释当前病历信息的关键性疑点

注意，只有诊断结果有足够证据支持时，才可以给出最终诊断；否则，必须继续辅助检查。


如果选择给出最终诊断，还需要根据病历对诊断进行扩充，最终输出两句话(核心诊断+完整诊断)，包括（仅在病历包含相关信息时描述，**严禁编造**）：
*   **a. 病因:** 明确指出导致疾病的直接病原体，以及根本的、促成的危险因素（例如，生活习惯、既往病史等）
*   **b. 核心疾病:** 陈述核心诊断名称
*   **c. 主要并发症:** 列出由核心疾病直接引发的最重要的并发症和伴随疾病
*   **d. 关键病理生理状态:** 描述这些并发症导致的身体功能异常状态


输出格式要求：
- 最终结果只包含是否继续检查或最终的完整诊断：**严禁**提供任何治疗方案、用药建议或健康指导，也不要输出病人的检查结果等无矛盾信息
- 如果选择继续辅助检查，输出 `继续辅助检查`
- 如果选择给出最终诊断，输出核心诊断+完整诊断，形如 `您的诊断结果为：xxx(核心诊断)\n您的完整诊断如下：xxx(完整诊断)`
- 先给出逐步的分析，然后最终结果放在<answer>标签中

输出示例1
逐步的分析...
<answer>
继续辅助检查
</answer>

输出示例2
...（逐步的分析）
<answer>
您的诊断结果为：颅内静脉窦血栓
您的完整诊断如下：由终末期肝病（肝硬化失代偿期）继发的凝血功能障碍和高氨血症所引发的颅内静脉窦血栓；并继发颅内高压、脑缺血性损伤及肝性脑病。
</answer>
"""

USER_PROMPT_DECISION_ACTION_AGENT = """患者现有的病历信息：
<病历>
{}
</病历>


初步鉴别诊断列表：
<诊断>
{}
</诊断>


请你逐步分析，判断医生下一步应该继续辅助检查，还是给出最终诊断。
"""


SYSTEM_PROMPT_AUXILIARY_EXAM_ACTION_AGENT = """你是一名负责辅助检查决策的临床医生。当前不能最终诊断，请根据病历和结构化鉴别诊断，开具下一步的辅助检查。

要求：
- 每轮最多 5 项，至少 1 项。
- 每行只能写 1 个具体检查项目，禁止把多个项目合并在同一行。
- 检查名称必须具体；CT/MRI/超声等需写明部位和方式。
- 避免重复已经做过且结果明确的检查。
- 不要输出治疗建议、诊断结论或检查结果。
- 需要考虑成本和效率，即病人需要支付的费用、等待的时间、是否有侵入性等。

输出格式：
先给出简短分析，然后把动作放在 <answer> 标签内，格式必须是：
<answer>
请求进行以下辅助检查：
- 检查1
- 检查2
</answer>
"""

USER_PROMPT_AUXILIARY_EXAM_ACTION_AGENT = """患者现有的病历信息：
<病历>
{}
</病历>

结构化鉴别诊断：
<诊断>
{}
</诊断>

请基于以上信息开具下一步辅助检查。
"""


class DoctorAgent_Diagnosis_Verify_Decision_Action(BaseDoctorAgent):
    """
    结构化鉴别诊断 -> 验证诊断 -> 判断是否诊断 -> 选择并生成下一步动作。
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

    # 处理一轮病人回复，并生成医生的下一步响应。
    def inference(self, patient_answer: str) -> str:
        self.current_summary += f"Patient: {patient_answer}\n"

        diagnosis_str = self._query(
            USER_PROMPT_LONG_DIAGNOSIS.format(self.current_summary),
            role="Doctor: differential diagnosis agent",
            system_prompt=SYSTEM_PROMPT_LONG_DIAGNOSIS,
        )
        self.diagnosis_str = self._verify_diagnoses(diagnosis_str)
        self.top3_diagnosis = self._top_diagnosis_names(self.diagnosis_str, top_k=3)

        if patient_answer in [Warning_from_too_many_auxiliary_exam]:
            answer = self._final_diagnosis_from_top1()
        else:
            decision_or_action = self._query(
                USER_PROMPT_DECISION_ACTION_AGENT.format(
                    self.current_summary, self.diagnosis_str
                ),
                role="Doctor: decision action agent",
                system_prompt=SYSTEM_PROMPT_DECISION_ACTION_AGENT,
            )
            if "您的诊断结果为：" in decision_or_action:
                answer = decision_or_action
                if "您的完整诊断如下：" not in answer:
                    answer += (
                        f"\n您的完整诊断如下：{self.top3_diagnosis[0] or '待明确诊断'}"
                    )
            else:
                action_type = "辅助检查"
                answer = self._make_action(action_type, self.diagnosis_str)

        self.current_summary += f"Doctor:{answer}\n"
        return answer

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

    # 根据当前诊断结果生成下一步检查动作。
    def _make_action(self, action_type: str, diagnosis_str: str) -> str:
        prompts = {
            "辅助检查": (
                USER_PROMPT_AUXILIARY_EXAM_ACTION_AGENT.format(
                    self.current_summary, diagnosis_str
                ),
                SYSTEM_PROMPT_AUXILIARY_EXAM_ACTION_AGENT,
                "Doctor: auxiliary exam action agent",
                "请求进行以下辅助检查：\n- 全血细胞计数\n- C-反应蛋白",
            ),
        }
        prompt, system_prompt, role, fallback = prompts[action_type]
        for _ in range(3):
            answer = self._query(prompt, role, system_prompt)
            if self._valid_action(answer, action_type):
                return answer
        return fallback

    # 逐个验证鉴别诊断，并补全概率和证据字段。
    def _verify_diagnoses(self, diagnosis_str: str) -> str:
        diagnosis_items = self._diagnosis_items(diagnosis_str)
        if not diagnosis_items:
            return (diagnosis_str or "").strip()

        verified_items = []
        for diagnosis_item in diagnosis_items:
            verified_str = self._query(
                USER_PROMPT_DIAGNOSIS_VERIFIER.format(
                    self.current_summary, diagnosis_item["诊断名称"]
                ),
                role="Doctor: diagnosis verifier",
                system_prompt=SYSTEM_PROMPT_DIAGNOSIS_VERIFIER,
            )
            verified_item = self._json_object(verified_str)
            if isinstance(verified_item, dict):
                verified_items.append(
                    self._normalize_verified_diagnosis(diagnosis_item, verified_item)
                )
            else:
                verified_items.append(
                    {
                        "诊断名称": diagnosis_item["诊断名称"],
                        "概率等级": "低",
                        "支持证据": "暂无明确支持证据",
                        "反对/矛盾证据": "暂无明确反对证据",
                        "待补充证据": "需进一步补充相关问诊、体征或检查证据",
                    }
                )

        return json.dumps({"鉴别诊断": verified_items}, ensure_ascii=False)

    # 检查动作输出是否符合格式要求。
    def _valid_action(self, answer: str, action_type: str) -> bool:
        if not answer:
            return False
        if action_type == "辅助检查":
            exam_lines = [
                line for line in answer.splitlines() if line.strip().startswith("-")
            ]
            return (
                answer.startswith("请求进行以下辅助检查：")
                and 1 <= len(exam_lines) <= 5
            )
        return False

    # 在检查次数超限时，用当前首位诊断生成最终诊断。
    def _final_diagnosis_from_top1(self) -> str:
        diagnosis = self.top3_diagnosis[0] or "待明确诊断"
        return f"您的诊断结果为：{diagnosis}\n您的完整诊断如下：{diagnosis}"

    # 从鉴别诊断 JSON 中提取诊断名称列表。
    def _diagnosis_items(self, diagnosis_str: str) -> list:
        diagnosis_json = self._json_object(diagnosis_str)
        if not isinstance(diagnosis_json, dict):
            return []

        diagnosis_items = []
        for item in diagnosis_json.get("鉴别诊断", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("诊断名称", "")).strip()
            if name:
                diagnosis_items.append({"诊断名称": name})
        return diagnosis_items

    # 规范 verifier 输出，保证后续代理收到稳定字段。
    def _normalize_verified_diagnosis(
        self, diagnosis_item: dict, verified_item: dict
    ) -> dict:
        probability = str(verified_item.get("概率等级", "")).strip()
        return {
            "诊断名称": diagnosis_item["诊断名称"],
            "概率等级": probability if probability in {"高", "中", "低"} else "低",
            "支持证据": str(
                verified_item.get("支持证据", "暂无明确支持证据")
            ).strip(),
            "反对/矛盾证据": str(
                verified_item.get("反对/矛盾证据", "暂无明确反对证据")
            ).strip(),
            "待补充证据": str(
                verified_item.get("待补充证据", "需进一步补充相关问诊、体征或检查证据")
            ).strip(),
        }

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

    # 返回当前保存的前三个诊断名称。
    def get_top3_diagnosis(self) -> list:
        return self.top3_diagnosis
