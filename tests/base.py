"""功能测试基类: 所有测试通过 OpenAI 兼容 API 调用。"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime


class FunctionalTestBase(ABC):
    """功能测试基类。

    所有测试共享:
    - OpenAI 兼容 API 连接 (sglang / vllm / trtllm 均支持)
    - 统一的结果输出格式
    - 日志保存到 session 目录
    """

    # 子类必须设置
    test_name: str = ""
    test_desc: str = ""

    def __init__(self, backend, num_samples: int | None = None):
        self.backend = backend
        self.num_samples = num_samples
        self.host = "127.0.0.1"
        self.port = backend.port
        self.model = backend.model_name
        self.base_url = f"http://{self.host}:{self.port}/v1"

        # 延迟导入, 仅功能测试需要
        try:
            from openai import OpenAI
        except ImportError:
            print("[Test] ✗ 需要安装 openai: pip install openai")
            sys.exit(1)

        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

    @abstractmethod
    def run(self) -> dict:
        """运行测试, 返回结果字典。

        Returns:
            {
                "test_name": str,
                "test_desc": str,
                "total": int,
                "passed": int,
                "failed": int,
                "accuracy": float,
                "details": list[dict],  # 每个用例的详细结果
                "duration": float,
            }
        """

    def execute(self, output_dir: str | None = None) -> dict:
        """执行测试并打印/保存结果。"""
        print(f"\n{'#'*70}")
        print(f"  功能测试: [{self.test_name}] {self.test_desc}")
        print(f"  模型: {self.model}")
        print(f"  API: {self.base_url}")
        if self.num_samples:
            print(f"  样本数: {self.num_samples}")
        print(f"{'#'*70}\n")

        start = time.time()
        result = self.run()
        result["duration"] = round(time.time() - start, 2)
        result["test_name"] = self.test_name
        result["test_desc"] = self.test_desc

        self._print_results(result)

        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _print_results(self, result: dict):
        """打印测试结果。"""
        total = result.get("total", 0)
        passed = result.get("passed", 0)
        failed = total - passed
        accuracy = (passed / total * 100) if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"  [{self.test_name}] 测试结果")
        print(f"{'='*60}")
        print(f"  总数: {total}")
        print(f"  通过: {passed}")
        print(f"  失败: {failed}")
        print(f"  准确率: {accuracy:.1f}%")
        print(f"  耗时: {result.get('duration', 0):.1f}s")
        print(f"{'='*60}")

        # 打印详细结果
        details = result.get("details", [])
        for i, d in enumerate(details, 1):
            status = "✓" if d.get("passed") else "✗"
            label = d.get("label", f"#{i}")
            print(f"  {status} {label}")
            if not d.get("passed") and d.get("error"):
                print(f"    └─ {d['error']}")

    def _save_results(self, result: dict, output_dir: str):
        """保存结果到 JSON 文件。"""
        test_dir = os.path.join(output_dir, f"test_{self.test_name}")
        os.makedirs(test_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(test_dir, f"result_{timestamp}.json")

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n[Test] 结果已保存: {result_path}")
