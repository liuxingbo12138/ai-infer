"""MMLU 多任务语言理解测试。

从 HuggingFace datasets 加载 MMLU 测试集,
构造选择题 prompt, 验证模型选择是否正确。
"""

import re
import sys

from .base import FunctionalTestBase

DEFAULT_NUM_SAMPLES = 50

CHOICES = ["A", "B", "C", "D"]


class MMLUTest(FunctionalTestBase):

    test_name = "mmlu"
    test_desc = "MMLU 多任务语言理解准确率测试"

    def __init__(self, backend, num_samples: int | None = None):
        super().__init__(backend, num_samples)
        self.num_samples = num_samples or DEFAULT_NUM_SAMPLES

    def _load_dataset(self):
        """加载 MMLU 数据集。"""
        try:
            from datasets import load_dataset
        except ImportError:
            print("[Test] ✗ 需要安装 datasets: pip install datasets")
            sys.exit(1)

        print(f"[Test] 加载 MMLU 数据集 (取前 {self.num_samples} 题)...")
        ds = load_dataset("cais/mmlu", "all", split="test")
        return list(ds.select(range(min(self.num_samples, len(ds)))))

    @staticmethod
    def _build_prompt(item: dict) -> str:
        """构造选择题 prompt。"""
        question = item["question"]
        choices = item["choices"]

        prompt = f"请回答以下选择题，直接给出答案字母（A/B/C/D）。\n\n"
        prompt += f"问题: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{CHOICES[i]}. {choice}\n"
        prompt += "\n请只回答一个字母（A、B、C 或 D）:"

        return prompt

    @staticmethod
    def _extract_choice(text: str) -> str | None:
        """从模型回复中提取选项字母。"""
        text = text.strip()

        # 模式1: 直接以字母开头
        if text and text[0].upper() in CHOICES:
            return text[0].upper()

        # 模式2: 找第一个独立的 A/B/C/D
        match = re.search(r"\b([A-D])\b", text.upper())
        if match:
            return match.group(1)

        # 模式3: 答案是/选/Answer is 后面的字母
        match = re.search(r"(?:答案[是为：:]*\s*|[Aa]nswer\s*(?:is)?\s*[：:]?\s*)([A-D])", text.upper())
        if match:
            return match.group(1)

        return None

    def run(self) -> dict:
        data = self._load_dataset()

        details = []
        passed_count = 0

        for i, item in enumerate(data):
            prompt = self._build_prompt(item)
            expected_idx = item["answer"]  # 0-3 的整数
            expected = CHOICES[expected_idx]
            subject = item.get("subject", "unknown")

            label = f"Q{i+1} ({subject})"
            print(f"  ▶ [{label}] {item['question'][:60]}...")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个知识渊博的助手。请直接回答选择题的答案字母。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=32,
                )

                model_reply = response.choices[0].message.content or ""
                model_choice = self._extract_choice(model_reply)

                passed = model_choice == expected

                if passed:
                    passed_count += 1
                    print(f"    ✓ {model_choice}")
                else:
                    print(f"    ✗ 期望: {expected}, 模型: {model_choice} (回复: {model_reply[:50]})")

                details.append({
                    "label": label,
                    "passed": passed,
                    "expected": expected,
                    "model_choice": model_choice,
                    "subject": subject,
                    "question": item["question"][:100],
                })

            except Exception as e:
                print(f"    ✗ 请求异常: {e}")
                details.append({
                    "label": label,
                    "passed": False,
                    "error": f"请求异常: {e}",
                    "subject": subject,
                    "question": item["question"][:100],
                })

        return {
            "total": len(data),
            "passed": passed_count,
            "accuracy": passed_count / len(data) if data else 0,
            "details": details,
        }
