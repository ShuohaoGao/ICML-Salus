from .call_LLM import LLM_Caller_for_One_Thread
from .data_loader import Scenario_OSCE
from .tools import batch_process_parallel, extract_label


Warning_from_too_many_auxiliary_exam = "来自系统的警告：你已经请求了太多次辅助检查，请不要再继续请求辅助检查了。现在请开始给出完整的最终诊断。"

prompt_makeup_one_exam_item = """你是一位资深临床医学专家，任务是**根据提供的病历，输出对应辅助检查的结果**。


## 处理规则
1. **判断检查结果是否在病历中明确记录**
   - 若病历中**已明确记录**该检查的**完整结果**，则直接输出病历中所载内容。
   - 若病历中**未明确记录**该检查结果，或者只有一部分记录而**缺少该检查的部分结果**，则需依据病历中的其他临床信息进行合理推断，给出符合临床实际的检查结果。
     - 如为数值型结果，应先推断其合理范围，再在该范围内进行采样，得出具体数值。
     - 所有数值型结果均需**额外标注是否异常**，例如使用“↑”表示升高、“↓”表示降低；如数值在正常参考范围内，则标注“（正常）”。

2. **禁止在结果中透露任何诊断性信息**，例如“符合××疾病表现”或“根据××疾病推断”。所输出的检查结果应模拟真实检查所得，保持客观、中立；最终结果不应表现出该结果是编造的。

3. **注重可读性**：如果有不常见的英文缩写，需要进行给出中文全称，例如：
  - 病历记录了“胰腺：头部见1mm低密度灶，与既往IPMN一致”
  - 需要给出“IPMN”的中文全称：“胰腺：头部见1mm低密度灶，与既往IPMN(IPMN, 导管内乳头状黏液性肿瘤)一致”

4. **先说异常**：方便医生快速阅读到有用信息。
    - 如果一项检查有多个指标/方面，则**优先描述异常发现**，再描述正常或未见异常的部分，方便阅读。
    - 如果检查结果无任何异常，则先说一句“检查结果无异常”，之后再描述具体的检查结果。

5. **拒绝回答多个检查项目**：如果医生请求了多个检查项目，则直接拒绝提供检查结果，并输出 `<hit>no</hit><answer>禁止把多个检查项目放在同一行</answer>`


## 输出格式
先给出逐步的分析，然后把结果输出中在xml标签内，是否命中的结果放在<hit>内，检查结果放在<answer>内。即：
逐步的分析
<hit>只要检查项目在病历中有明确记录或有部分记录，则为yes；否则没有任何相关的记录为no。此处只能是yes或者no，不要有其他文字</hit>
<answer>
检查名称（需要和给定的名称保持一致）：检查结果（所有检查的结果放在一行当中）
</answer>


**输出示例1**：假设需要响应`尿常规和尿培养`的检查结果，并且病历没有记录该检查的结果
...（逐步的分析）
<hit>no</hit>
<answer>
尿常规和尿培养：葡萄糖 3+ ↑，尿试纸酮体 1+ ↑，大肠埃希菌 1.2×10^5 CFU/mL ↑（参考值 <10^4 CFU/mL）；尿蛋白阴性，尿沉渣镜检红细胞、白细胞正常。
</answer>


**输出示例2**：假设需要响应`腹部超声`的检查结果，并且病历没有记录该检查的结果
...（逐步的分析）
<hit>no</hit>
<answer>
腹部超声：检查结果无异常。肝脏、胆囊、胰腺、脾脏及双肾形态大小正常，未见占位性病变。
</answer>

完整病历：
<病历>
{}
</病历>


需响应的辅助检查项目如下：
{}



请先给出逐步的分析，然后按格式输出结果。"""


def makeup_one_exam_result(
    item: str,
    scenario: Scenario_OSCE,
    model_name: str,
    LLM_caller: LLM_Caller_for_One_Thread = None,
):
    def build_result(answer: str, hit: int, hit_judged: int, log_list: list = None) -> dict:
        return {
            "answer": answer,
            "hit": hit,
            "hit_judged": hit_judged,
            "log_list": log_list or [],
        }

    if item.startswith("-"):
        item = item[1:].strip()
    if item in scenario.tests.get("extra", {}):
        ans = scenario.tests["extra"][item]
        if ans and isinstance(ans, str):
            return build_result(ans, 1, 1)

    patient_record = scenario.record_for_auxiliary_exam_agent()
    if not LLM_caller:
        LLM_caller = LLM_Caller_for_One_Thread()
    start_log_idx = len(LLM_caller.LLM_log_list)
    for _ in range(2):
        try:
            response = LLM_caller.query_model(
                model_str=model_name,
                prompt=prompt_makeup_one_exam_item.format(patient_record, item),
                system_prompt=None,
                role="MedTestAgent",
                ensure_label="answer",
            )
            ans = extract_label(response, "answer")
            hit = extract_label(response, "hit")
            hit = True if hit and hit.lower() == "yes" else False
            if ans:
                return build_result(
                    f"- {ans}",
                    int(hit),
                    1,
                    LLM_caller.LLM_log_list[start_log_idx:],
                )
        except:
            pass
    return build_result(
        f"- {item}：结果丢失，请重新申请此检查。",
        0,
        0,
        LLM_caller.LLM_log_list[start_log_idx:],
    )


class MedTestAgent:
    """
    Provides medical test results to the doctor.
    """

    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        scenario: Scenario_OSCE,
        model_name="gpt",
    ) -> None:
        self.scenario = scenario
        self.model_name = model_name
        self.LLM_caller = LLM_caller
        self.last_hit_cnt = 0
        self.last_hit_judged_cnt = 0

    def inference(self, question: str) -> str:
        item_list = [x for x in question.split("\n") if x.startswith("-")]
        if len(item_list) == 0:
            return "来自系统的警告：请你严格按照格式重新输出。"
        if len(item_list) > 5:
            return "来自系统的警告：一轮最多请求5项辅助检查。"
        answer_of_each_item = batch_process_parallel(
            func=makeup_one_exam_result,
            args_list=[
                [item, self.scenario, self.model_name, self.LLM_caller]
                for item in item_list
            ],
            num_processes=len(item_list),
            use_thread=True,
        )
        for item in answer_of_each_item:
            self.LLM_caller.LLM_log_list.extend(item["log_list"])
        self.last_hit_cnt = sum(item["hit"] for item in answer_of_each_item)
        self.last_hit_judged_cnt = sum(item["hit_judged"] for item in answer_of_each_item)
        return "\n".join(item["answer"] for item in answer_of_each_item)
