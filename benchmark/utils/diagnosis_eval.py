from .call_LLM import LLM_Caller_for_One_Thread


SYSTEM_PROMPT_judge_diagnosis_correct_level = """你是一名医学专家，负责阅卷：基于提供的标准答案，对临床医生的诊断进行精准、客观的评估。
你必须严格遵循以下的**评估流程**来给出判断和解释。


# 评估流程
你必须严格遵守以下两个步骤进行评估：

## 第一步：核心诊断解构
在评估之前，你必须先分析【最终诊断】，将其拆解为：
1.  **核心诊断**：
    *   导致患者当前就诊的主要原因，是诊断中比较严重的若干个疾病。
    *   包括但不限于危及生命或需要紧急医疗干预的疾病。
    *   *评估原则*：后续评估时，这是主要依据。
2.  **次要诊断**：
    *   可以是长期稳定的慢性基础病（如控制良好的高血压），通常不会危及生命，也不需要紧急干预
    *   也可以是核心诊断所直接导致的伴随疾病和伴随症状。
    *   *评估原则*：后续评估时，通常忽略不计。如果医生遗漏了次要诊断，**不**扣分。

## 第二步：基于治疗后果的四级评估
基于“核心诊断”，对比“医生诊断”，根据**对治疗核心诊断的主要方案的影响**选择以下四个等级之一：

### 1. 完全正确
*   **定义**：医生诊断准确覆盖了所有的**核心诊断**。
*   **特征**：
    *   病因、部位、性质完全一致。
    *   仅遗漏了次要诊断。

### 2. 临床可接受
*   **定义**：医生诊断在**核心方向**上正确，仅在**精度**上存在瑕疵。
*   **关键特征**：
    *   **精度瑕疵**：使用了正确的上位概念（如写了“细菌性肺炎”而非“肺炎链球菌肺炎”），或分型/分级不精准。
*   **判别金标准**：**后续治疗方案不会有太大区别**。医生基于该诊断开具的处方，涵盖了患者所需的核心治疗（如抗生素覆盖面正确、手术方式基本一致）。

### 3. 部分正确
*   **定义**：医生诊断的大方向（如系统或症状）有一定的道理，但存在**关键性缺失或模糊**，导致治疗方案存在明显缺陷。
*   **关键特征**：
    *   **核心遗漏**：并存多个核心诊断中，漏掉了其中一个
    *   **精度过于模糊**：诊断过于笼统，导致无法进行必须的特异性治疗（如“感染性休克”未指明感染源，导致抗生素滥用或无效）;或者只给出了核心诊断导致的严重综合征（如休克、心衰），但未诊断原发病。
    *   **分型错误致治疗改变**：例如，将需要手术的类型误判为保守治疗类型。
*   **判别金标准**：**治疗方案有明显区别**，或者**遗漏核心诊断**。虽然医生没有完全误诊（**有一定的道理**），但患者无法获得针对性的关键治疗（如漏了溶栓药、漏了特定手术）。

### 4. 不正确
*   **定义**：**完全错误**的诊断，不仅没有覆盖到任何核心诊断，也没有参考价值。
*   **特征**：将A病误诊为B病（机制完全不同），或仅诊断“腹痛”、“待查”。
*   **判别金标准**：**治疗完全错误或延误**。


# 注意
1. 次要诊断，相比于核心诊断，更加次要；次要诊断不依赖于具体的疾病，例如“脂肪肝”本身可以作为核心诊断，但是如果病人同时存在“肺癌”，那么“肺癌”就是核心诊断，而“脂肪肝”则成为次要诊断。
2. **治疗方案只考虑针对核心诊断的大致诊断路径**，不考虑次要诊断，也无需考虑详细具体的治疗方案。
3. 相比于完全正确，如果在精度上存在错误，那么有可能被分类到“临床可接受”或“部分正确”，具体分到哪个类别，需要根据后续治疗方案判断这种错误程度是否可以接受。
4. 如果发现医生试图罗列多个诊断来蒙答案，或者多个诊断之间存在矛盾，则需要分类为“不正确”或者“部分正确”。
5. 采取严格阅卷方式：**如果在2个级别之间犹豫不决，则优先选择较低级别**。例如，不确定是“临床可接受”还是“部分正确”时，直接选择“部分正确”。


# 回复格式
逐步的分析过程
<answer>判断标签</answer>

其中，判断标签必须是4个类别中的一个，例如
<answer>临床可接受</answer>
"""

USER_PROMPT_judge_diagnosis_correct_level = """{}


请基于提供的医疗信息，对临床医生的诊断进行评估并分类。先给出逐步的分析，并把最终判断放在标签<answer></answer>内。"""

prompt_diagnosis_from_record = """你是一名医学专家，你的任务是根据提供的患者病历，推理出top-5鉴别诊断列表。

**核心要求：**
- **诊断排序**：诊断列表按可能性由高到低排列。
- **诊断完整性**：每个诊断都应是完整的，应具体明确（例如：使用“右下叶肺炎”而非“肺炎”；“冠心病不稳定型心绞痛”而非“心脏病”）；可包含主要疾病和相关的并发症/合并症（例如：2型糖尿病 合并 社区获得性肺炎）。
- **诊断竞争性**：列表中的各项诊断应该是相互竞争的备选方案（即鉴别诊断）。**不要将一个统一病理过程的不同方面拆分成独立的条目**（如将“社区获得性肺炎”和“发热”分别列为两个诊断）。
- **聚焦诊断**：你的回答应专注于诊断推理过程和最终的诊断列表。**严禁**提供任何治疗方案、用药建议或健康指导，也不要包含病人的检查结果等信息。
- **诊断个数**：允许鉴别诊断个数不足5个。

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


以下是病人的信息：
<病历>
{}
</病历>


现在请先给出逐步的分析，然后输出若干个相互竞争、完整的诊断方案，不要给出其他无内容。
"""


def judge_correct_diagnosis_level(
    correct_diagnosis: str,
    diagnosis_to_be_judged: str,
    LLM_caller: LLM_Caller_for_One_Thread,
    other_info=None,
    model_name: str = "GLM-Z1-Flash",
    level_list: list = ["完全正确", "临床可接受", "部分正确", "不正确"],
) -> tuple[bool, str]:
    if not diagnosis_to_be_judged or not correct_diagnosis:
        return False, level_list[-1]
    try:
        if not other_info:
            other_info = "空"
        prefix = (
            f"最终诊断 (绝对正确的标准答案)：\n{correct_diagnosis}\n\n"
            + f"医生诊断 (有待评估):\n{diagnosis_to_be_judged}\n\n"
        )
        for _ in range(3):
            ans = LLM_caller.query_model_and_extract_label(
                model_str=model_name,
                prompt=USER_PROMPT_judge_diagnosis_correct_level.format(prefix),
                system_prompt=SYSTEM_PROMPT_judge_diagnosis_correct_level,
                role="Judge Diagnosis Level",
                ensure_label="answer",
            )
            if ans in level_list:
                correctness = ans in ["完全正确", "临床可接受"]
                return correctness, ans
    except Exception as e:
        print("Error in judge_correct_diagnosis_level:", e, flush=True)
    return False, level_list[-1]


def diagnose_from_record(
    record: str, model_name: str, LLM_caller: LLM_Caller_for_One_Thread = None
) -> str:
    try:
        if LLM_caller is None:
            LLM_caller = LLM_Caller_for_One_Thread()
        diagnosis_top5 = LLM_caller.query_model_and_extract_label(
            model_str=model_name,
            prompt=prompt_diagnosis_from_record.format(record),
            system_prompt=None,
            role="Doctor",
            ensure_label="answer",
        )
        diagnosis = diagnosis_top5.split("\n")[0]
        return diagnosis
    except:
        return None


def diagnose_from_record_and_judge_correctness(
    record: str,
    correct_diagnosis: str,
    model_name: str,
    LLM_caller: LLM_Caller_for_One_Thread = None,
    other_info: str = "",
) -> str:
    if LLM_caller is None:
        LLM_caller = LLM_Caller_for_One_Thread()
    diagnosis = diagnose_from_record(
        record=record, model_name=model_name, LLM_caller=LLM_caller
    )
    correctness, level = judge_correct_diagnosis_level(
        correct_diagnosis=correct_diagnosis,
        diagnosis_to_be_judged=diagnosis,
        LLM_caller=LLM_caller,
        other_info=other_info,
        model_name=model_name,
    )
    return {"diagnosis": diagnosis, "correctness": correctness, "level": level}, LLM_caller.LLM_log_list


def judge_topk(
    topk_diagnosis: list,
    correct_diagnosis: str,
    model_name: str,
    LLM_caller: LLM_Caller_for_One_Thread = None,
    other_info: str = "",
):
    correctness_list = []
    if LLM_caller is None:
        LLM_caller = LLM_Caller_for_One_Thread()
    for d in topk_diagnosis:
        if d is None:
            correctness_list.append(
                {"diagnosis": None, "correctness": False, "level": "不正确"}
            )
        else:
            correctness, level = judge_correct_diagnosis_level(
                correct_diagnosis=correct_diagnosis,
                diagnosis_to_be_judged=d,
                LLM_caller=LLM_caller,
                model_name=model_name,
                other_info=other_info,
            )
            correctness_list.append(
                {"diagnosis": d, "correctness": correctness, "level": level}
            )
    return correctness_list, LLM_caller.LLM_log_list
