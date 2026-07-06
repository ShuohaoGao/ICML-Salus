import re
from utils.tools import extract_label
from .interface import BaseDoctorAgent
from utils.call_LLM import LLM_Caller_for_One_Thread

#########################################################################################
# 全流程医生Agent

SYSTEM_PROMPT_inquiry_examination_doctor_agent = """你是一名临床医生，需要通过多轮交互完成诊断。

你需要通过多轮的辅助检查，最终给出诊断。

要求：
- 证据充分时再给最终诊断；证据不足时继续收集信息
- 如果请求辅助检查，每一轮最多同时请求5项辅助检查

输出格式：
先给出简短分析，再把最终动作放在 `<answer></answer>` 中。


如果是辅助检查，输出格式：
```
请求进行以下辅助检查：
- 检查1
- 检查2
- ...
```

如果是最终诊断，输出格式：
```
您的诊断结果为：...
您的完整诊断如下：...
鉴别诊断列表：
1. top-1诊断
2. top-2诊断
3. top-3诊断
```

示例1：
<answer>
请求进行以下辅助检查：
- 全血计数
- C-反应蛋白
</answer>


示例2：
<answer>
您的诊断结果为：镰状细胞病合并急性血管闭塞危象、溶血性贫血和股静脉血栓形成
您的完整诊断如下：由镰状细胞病（血红蛋白S分子聚合）及脾脏切除术后状态（血栓形成倾向）共同导致的急性血管闭塞危象；并继发溶血性贫血、右侧股静脉血栓形成及右侧肢体急性神经功能缺损（中枢神经系统受累）。
鉴别诊断列表：
1. 镰状细胞病合并急性血管闭塞危象、溶血性贫血和股静脉血栓形成
2. 遗传性球形红细胞增多症合并再障危象和血栓形成
3. 抗磷脂综合征合并灾难性血管闭塞和微血管病性溶血
</answer>
"""


class Doctor_1Agent(BaseDoctorAgent):
    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        model_name="gpt4",
    ) -> None:
        super().__init__(LLM_caller, model_name)
        self.messages_history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_inquiry_examination_doctor_agent,
            }
        ]

    def inference(self, patient_answer: str) -> str:
        """
        question: this is from the response of patient of other measurement agents
        """
        self.messages_history.append({"role": "user", "content": patient_answer})
        raw_answer = self.LLM_caller.query_model(
            model_str=self.model_name,
            messages=self.messages_history,
            role="Doctor",
            ensure_label="answer",
        )
        extracted_ans = extract_label(raw_answer, "answer")
        if extracted_ans:
            answer = extracted_ans
        else:
            answer = raw_answer
        self.messages_history.append({"role": "assistant", "content": raw_answer})
        if "鉴别诊断列表" in answer:
            split_result = re.split(r"鉴别诊断列表[:：]\s*", answer, maxsplit=1)
            if len(split_result) == 2:
                answer, diag_list_str = split_result[0].strip(), split_result[1].strip()
            else:
                diag_list_str = answer.split("鉴别诊断列表", 1)[-1].strip()
                answer = answer.split("鉴别诊断列表", 1)[0].strip()
            diag_lines = diag_list_str.splitlines()
            top3 = []
            for line in diag_lines:
                line = line.strip()
                if line:
                    diag = re.sub(r"^[-*]\s*", "", line)
                    diag = re.sub(r"^\d+[\.\)\s]+", "", diag).strip()
                    top3.append(diag)
                if len(top3) >= 3:
                    break
            self.top3_diagnosis = (top3 + [None] * 3)[:3]
        return answer
