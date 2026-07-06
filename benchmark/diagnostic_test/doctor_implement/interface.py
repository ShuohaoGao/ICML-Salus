from utils.call_LLM import LLM_Caller_for_One_Thread
from abc import ABC, abstractmethod


class BaseDoctorAgent(ABC):
    """
    医生Agent的基类。
    封装了通用的初始化、对话摘要更新逻辑。
    """

    def __init__(
        self,
        LLM_caller: LLM_Caller_for_One_Thread,
        model_name="gpt4",
    ) -> None:
        self.model_name = model_name
        self.LLM_caller = LLM_caller
        self.top3_diagnosis = [None] * 3

    @abstractmethod
    def inference(self, patient_answer: str) -> str:
        pass

    def get_top3_diagnosis(self) -> list:
        return self.top3_diagnosis