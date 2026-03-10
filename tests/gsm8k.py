"""GSM8K 数学推理测试。

从 HuggingFace datasets 加载 GSM8K 测试集,
向模型发送数学题并验证最终数值答案是否正确。
"""

import re
import sys

from .base import FunctionalTestBase

DEFAULT_NUM_SAMPLES = 100


class GSM8KTest(FunctionalTestBase):

    test_name = "gsm8k"
    test_desc = "GSM8K 数学推理准确率测试"

    def __init__(self, backend, num_samples: int | None = None):
        super().__init__(backend, num_samples)
        self.num_samples = num_samples or DEFAULT_NUM_SAMPLES

    def _load_dataset(self):
        """加载 GSM8K 数据集。"""
        try:
            from datasets import load_dataset
        except ImportError:
            print("[Test] ✗ 需要安装 datasets: pip install datasets")
            sys.exit(1)

        print(f"[Test] 加载 GSM8K 数据集 (取前 {self.num_samples} 题)...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return list(ds.select(range(min(self.num_samples, len(ds)))))

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """从 GSM8K 标准答案中提取数值 (#### 后的数字)。"""
        match = re.search(r"####\s*(.+)", text)
        if match:
            return match.group(1).strip().replace(",", "")
        return None

    @staticmethod
    def _extract_model_answer(text: str) -> str | None:
        """从模型回复中提取最终数值答案。

        尝试多种模式:
        1. #### 格式
        2. \\boxed{} 格式
        3. 最后一个独立数字
        """
        # 模式1: #### 数字
        match = re.search(r"####\s*(.+)", text)
        if match:
            return match.group(1).strip().replace(",", "")

        # 模式2: \\boxed{数字}
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip().replace(",", "")

        # 模式3: 最后一行或最后出现的数字
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    def run(self) -> dict:
        data = self._load_dataset()

        details = []
        passed_count = 0

        for i, item in enumerate(data):
            question = item["question"]
            answer_text = item["answer"]
            expected = self._extract_answer(answer_text)

            label = f"Q{i+1}"
            print(f"  ▶ [{label}] {question[:60]}...")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个数学助手。请一步步解题，"
                                "最后用 #### 加数字的格式给出最终答案。"
                                "例如: #### 42"
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )

                model_reply = response.choices[0].message.content or ""
                model_answer = self._extract_model_answer(model_reply)

                passed = False
                if model_answer and expected:
                    # 标准化比较: 去除前导零、小数点
                    try:
                        passed = float(model_answer) == float(expected)
                    except ValueError:
                        passed = model_answer == expected

                if passed:
                    passed_count += 1
                    print(f"    ✓ 答案: {model_answer}")
                else:
                    print(f"    ✗ 期望: {expected}, 模型: {model_answer}")

                details.append({
                    "label": label,
                    "passed": passed,
                    "expected": expected,
                    "model_answer": model_answer,
                    "question": question[:100],
                })

            except Exception as e:
                print(f"    ✗ 请求异常: {e}")
                details.append({
                    "label": label,
                    "passed": False,
                    "error": f"请求异常: {e}",
                    "question": question[:100],
                })

        return {
            "total": len(data),
            "passed": passed_count,
            "accuracy": passed_count / len(data) if data else 0,
            "details": details,
        }
