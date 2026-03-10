"""结构化输出 (Structured Output) 功能测试。

验证模型是否能生成符合指定 JSON Schema 的输出:
- 输出为合法 JSON
- 符合 schema 定义的结构
- 必需字段齐全
"""

import json

from .base import FunctionalTestBase


# 内置测试用例: (prompt, json_schema, validator_func)
TEST_CASES = [
    {
        "label": "书籍推荐",
        "prompt": "推荐3本Python编程书籍，返回JSON格式，包含title和author字段",
        "json_schema": {
            "name": "book_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "books": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "author": {"type": "string"},
                            },
                            "required": ["title", "author"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["books"],
                "additionalProperties": False,
            },
        },
        "validate": lambda data: (
            isinstance(data.get("books"), list)
            and len(data["books"]) > 0
            and all(
                isinstance(b.get("title"), str) and isinstance(b.get("author"), str)
                for b in data["books"]
            )
        ),
    },
    {
        "label": "用户信息",
        "prompt": "生成一个虚拟用户的信息，包含姓名、年龄和邮箱",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
                "additionalProperties": False,
            },
        },
        "validate": lambda data: (
            isinstance(data.get("name"), str)
            and isinstance(data.get("age"), int)
            and isinstance(data.get("email"), str)
        ),
    },
    {
        "label": "城市信息",
        "prompt": "介绍一下上海这座城市，返回JSON包含city_name、country、population和description",
        "json_schema": {
            "name": "city_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string"},
                    "country": {"type": "string"},
                    "population": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["city_name", "country", "population", "description"],
                "additionalProperties": False,
            },
        },
        "validate": lambda data: (
            isinstance(data.get("city_name"), str)
            and isinstance(data.get("country"), str)
            and isinstance(data.get("population"), int)
            and isinstance(data.get("description"), str)
        ),
    },
    {
        "label": "分类结果",
        "prompt": "对以下文本进行情感分类（正面/负面/中性）：'这家餐厅的菜品非常好吃，服务也很周到'",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["正面", "负面", "中性"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["text", "sentiment", "confidence"],
                "additionalProperties": False,
            },
        },
        "validate": lambda data: (
            isinstance(data.get("text"), str)
            and data.get("sentiment") in ("正面", "负面", "中性")
            and isinstance(data.get("confidence"), (int, float))
        ),
    },
    {
        "label": "日程安排",
        "prompt": "帮我创建一个明天的日程安排，包含3个事项，每个事项有时间、标题和地点",
        "json_schema": {
            "name": "schedule",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string"},
                                "title": {"type": "string"},
                                "location": {"type": "string"},
                            },
                            "required": ["time", "title", "location"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["date", "events"],
                "additionalProperties": False,
            },
        },
        "validate": lambda data: (
            isinstance(data.get("date"), str)
            and isinstance(data.get("events"), list)
            and len(data["events"]) > 0
            and all(
                isinstance(e.get("time"), str)
                and isinstance(e.get("title"), str)
                and isinstance(e.get("location"), str)
                for e in data["events"]
            )
        ),
    },
]


class StructuredOutputTest(FunctionalTestBase):

    test_name = "structured_output"
    test_desc = "结构化输出功能测试"

    def run(self) -> dict:
        cases = TEST_CASES
        if self.num_samples and self.num_samples < len(cases):
            cases = cases[: self.num_samples]

        details = []
        passed_count = 0

        for case in cases:
            result = self._run_case(case)
            details.append(result)
            if result["passed"]:
                passed_count += 1

        return {
            "total": len(cases),
            "passed": passed_count,
            "accuracy": passed_count / len(cases) if cases else 0,
            "details": details,
        }

    def _run_case(self, case: dict) -> dict:
        label = case["label"]
        print(f"  ▶ [{label}] {case['prompt'][:50]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": case["prompt"]}],
                response_format={
                    "type": "json_schema",
                    "json_schema": case["json_schema"],
                },
            )

            content = response.choices[0].message.content or ""

            # 检查是否为合法 JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return {
                    "label": label,
                    "passed": False,
                    "error": f"输出非合法 JSON: {content[:100]}...",
                }

            # 结构验证
            validator = case.get("validate")
            if validator and not validator(data):
                return {
                    "label": label,
                    "passed": False,
                    "error": f"JSON 结构不匹配 schema",
                    "output": data,
                }

            preview = json.dumps(data, ensure_ascii=False)
            print(f"    ✓ {preview[:80]}...")
            return {
                "label": label,
                "passed": True,
                "output": data,
            }

        except Exception as e:
            return {
                "label": label,
                "passed": False,
                "error": f"请求异常: {e}",
            }
