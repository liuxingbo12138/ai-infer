from .base import BackendBase
from .sglang import SGLangBackend
from .vllm import VLLMBackend
from .trtllm import TRTLLMBackend

BACKEND_REGISTRY = {
    "sglang": SGLangBackend,
    "vllm": VLLMBackend,
    "trtllm": TRTLLMBackend,
}


def get_backend(name: str) -> type[BackendBase]:
    """根据名称获取后端类。"""
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"未知后端 '{name}', 可选: {available}")
    return cls
