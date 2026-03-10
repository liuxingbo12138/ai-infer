"""功能测试注册表。"""

from .tool_calling import ToolCallingTest
from .structured_output import StructuredOutputTest
from .gsm8k import GSM8KTest
from .mmlu import MMLUTest

TEST_REGISTRY = {
    "tool_calling": ToolCallingTest,
    "structured_output": StructuredOutputTest,
    "gsm8k": GSM8KTest,
    "mmlu": MMLUTest,
}

ALL_TEST_NAMES = sorted(TEST_REGISTRY.keys())


def get_test(name: str):
    """根据名称获取测试类。"""
    cls = TEST_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(ALL_TEST_NAMES)
        raise ValueError(f"未知测试类型 '{name}', 可选: {available}")
    return cls
