import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from doctor_implement.interface import BaseDoctorAgent
from utils.call_LLM import LLM_Caller_for_One_Thread


def _import_doctor_module(module_name: str):
    try:
        return importlib.import_module(f"doctor_implement.{module_name}")
    except ImportError:
        if __package__:
            return importlib.import_module(f"{__package__}.doctor_implement.{module_name}")
        raise


@lru_cache(maxsize=1)
def _discover_doctor_classes() -> dict[str, type[BaseDoctorAgent]]:
    doctor_dir = Path(__file__).resolve().parent / "doctor_implement"
    class_map = {}

    for file_path in doctor_dir.glob("*.py"):
        if file_path.stem in {"interface", "__init__"}:
            continue

        module = _import_doctor_module(file_path.stem)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not issubclass(cls, BaseDoctorAgent) or cls is BaseDoctorAgent:
                continue
            class_map[cls.__name__] = cls

    return class_map


def create_doctor_agent(
    class_name: str,
    LLM_caller: LLM_Caller_for_One_Thread,
    model_name: str = "gpt4",
) -> "BaseDoctorAgent":
    """
    根据类名字符串动态查找并创建医生Agent实例。

    Args:
        class_name (str): 目标类的名称，例如 "DoctorAgent_Diagnosis_Exam_Inherited"。

    Returns:
        BaseDoctorAgent: 对应类的实例。

    Raises:
        ValueError: 如果找不到指定的类名，或者该类不是BaseDoctorAgent的有效子类。
    """
    agent_class = _discover_doctor_classes().get(class_name)

    # --- 安全性校验 ---
    # 1. 检查是否找到了对应的名称，并且它确实是一个类
    if not agent_class or not inspect.isclass(agent_class):
        available_classes = sorted(_discover_doctor_classes().keys())
        raise ValueError(
            f"错误：未在 doctor_implement 目录中找到名为 '{class_name}' 的类。"
            + f" 当前可用类: {available_classes}"
        )

    # 2. 检查这个类是否是 BaseDoctorAgent 的子类（但不是基类本身）
    # 这是为了确保我们实例化的对象拥有正确的接口 (如 inference 方法)
    if not issubclass(agent_class, BaseDoctorAgent) or agent_class is BaseDoctorAgent:
        raise TypeError(f"错误：类 '{class_name}' 不是 BaseDoctorAgent 的有效子类。")

    # --- 实例化并返回 ---
    # 如果所有检查都通过，就安全地创建实例
    return agent_class(
        LLM_caller=LLM_caller,
        model_name=model_name,
    )
