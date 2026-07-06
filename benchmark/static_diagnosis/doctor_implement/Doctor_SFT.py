from .interface import BaseDoctorAgent, LLM_Caller_for_One_Thread


class Doctor_SFT(BaseDoctorAgent):
    """
    只适用于本课题的微调LLM；不可用于其他基于prompt的LLM
    """

    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        model_name="gpt4",
        max_auxiliary_num=8,
        step_by_step_verify=False,
    ) -> None:
        super().__init__(LLM_caller, model_name)
        self.max_auxiliary_num = max_auxiliary_num
        self.step_by_step_verify = step_by_step_verify
        # 用于记录对话
        self.conversation_history = []
        self.enable_long_diagnosis = False
        self.generation_kwargs = {
            "max_tokens": 7168,
            "temperature": 0.7,
        }

    def get_top3_diagnosis(self, record_summary: str) -> list:
        """
        只诊断，返回top-k Diagnosis
        """
        long_diagnosis_system_prompt = (
            "你是一名专业医生，负责根据提供的病人信息，进行鉴别诊断。"
        )
        diagnosis_user_prompt = "<病历>\n{}\n</病历>"

        verify_system_prompt = (
            "你负责验证指定的诊断是否与病人的信息矛盾。若无矛盾，输出yes"
        )
        verify_user_prompt = "<病历>\n{}\n</病历>\n\n<诊断>\n{}\n</诊断>"
        # long 鉴别诊断
        messages = [
            {"role": "system", "content": long_diagnosis_system_prompt},
            {"role": "user", "content": diagnosis_user_prompt.format(record_summary)},
        ]
        diagnosis_str = None
        for _ in range(3):
            raw_answer = self.LLM_caller.query_model(
                model_str=self.model_name,
                messages=messages,
                role="long diagnosis",
                ensure_label=None,
                generation_kwargs=self.generation_kwargs,
            )
            diagnosis_str = raw_answer.split("</reason>")[-1].strip()
            if diagnosis_str and len(diagnosis_str) < 800:
                break
            else:
                diagnosis_str = None
        if not diagnosis_str:
            raise ValueError("辅检时鉴别诊断失败")

        # 地毯式验证
        diagnosis_list = [d.strip() for d in diagnosis_str.split("\n") if d.strip()]
        verified_diagnosis_list = []
        wrong_diagnosis_list = []
        for d in diagnosis_list:
            messages = [
                {"role": "system", "content": verify_system_prompt},
                {
                    "role": "user",
                    "content": verify_user_prompt.format(record_summary, d),
                },
            ]
            for _ in range(3):
                if self.step_by_step_verify:
                    raw_answer = self.LLM_caller.query_model(
                        model_str=self.model_name,
                        messages=messages,
                        role="verify diagnosis",
                        ensure_label=None,
                        generation_kwargs=self.generation_kwargs,
                    )
                else:
                    raw_answer = "<reason>无需地毯式验证</reason>yes"
                verify_result = raw_answer.split("</reason>")[-1].strip()
                if verify_result:
                    if verify_result.startswith("yes"):
                        verified_diagnosis_list.append(d)
                    else:
                        wrong_diagnosis_list.append(d)
                    break

        # 记录最新的top-3
        self.top3_diagnosis = []
        if len(verified_diagnosis_list) > 0:
            self.top3_diagnosis = verified_diagnosis_list + wrong_diagnosis_list
            # 只保留3个
            self.top3_diagnosis = (self.top3_diagnosis + [None] * 3)[:3]
        else:
            # 都没通过验证
            self.top3_diagnosis = diagnosis_list
            # 只保留3个
            self.top3_diagnosis = (self.top3_diagnosis + [None] * 3)[:3]

        return self.top3_diagnosis
