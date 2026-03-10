"""工具调用 (Tool Calling) 功能测试。

验证模型是否能正确生成 tool_calls:
- 函数名匹配
- 参数为合法 JSON
- 必需参数存在
"""

import json

from .base import FunctionalTestBase


# 内置测试用例: (prompt, tools, expected_function, required_params)
TEST_CASES = [
    {
        "label": "天气查询",
        "prompt": "北京今天天气怎么样？",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "expected_function": "get_weather",
        "required_params": ["city"],
    },
    {
        "label": "数学计算",
        "prompt": "帮我算一下 123 乘以 456 等于多少",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式",
                            },
                        },
                        "required": ["expression"],
                    },
                },
            }
        ],
        "expected_function": "calculate",
        "required_params": ["expression"],
    },
    {
        "label": "网页搜索",
        "prompt": "搜索一下最新的 AI 新闻",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expected_function": "web_search",
        "required_params": ["query"],
    },
    {
        "label": "发送邮件",
        "prompt": "帮我给 alice@example.com 发一封邮件，主题是会议通知，内容是明天下午3点开会",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "发送电子邮件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string",
                                "description": "收件人邮箱",
                            },
                            "subject": {
                                "type": "string",
                                "description": "邮件主题",
                            },
                            "body": {
                                "type": "string",
                                "description": "邮件内容",
                            },
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            }
        ],
        "expected_function": "send_email",
        "required_params": ["to", "subject", "body"],
    },
    {
        "label": "多工具选择",
        "prompt": "帮我翻译一下 'Hello World' 到中文",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "translate",
                    "description": "翻译文本到指定语言",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "要翻译的文本",
                            },
                            "target_language": {
                                "type": "string",
                                "description": "目标语言",
                            },
                        },
                        "required": ["text", "target_language"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ],
        "expected_function": "translate",
        "required_params": ["text", "target_language"],
    },
]


class ToolCallingTest(FunctionalTestBase):

    test_name = "tool_calling"
    test_desc = "工具调用功能测试"

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
                tools=case["tools"],
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 检查是否有 tool_calls
            if not message.tool_calls:
                return {
                    "label": label,
                    "passed": False,
                    "error": "模型未生成 tool_calls",
                    "response": message.content or "",
                }

            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments

            # 检查函数名
            if func_name != case["expected_function"]:
                return {
                    "label": label,
                    "passed": False,
                    "error": f"函数名不匹配: 期望 {case['expected_function']}, 实际 {func_name}",
                }

            # 检查参数是否为合法 JSON
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                return {
                    "label": label,
                    "passed": False,
                    "error": f"参数非合法 JSON: {func_args_str[:100]}",
                }

            # 检查必需参数
            missing = [
                p for p in case["required_params"] if p not in func_args
            ]
            if missing:
                return {
                    "label": label,
                    "passed": False,
                    "error": f"缺少必需参数: {missing}",
                }

            print(f"    ✓ {func_name}({json.dumps(func_args, ensure_ascii=False)[:80]})")
            return {
                "label": label,
                "passed": True,
                "function": func_name,
                "arguments": func_args,
            }

        except Exception as e:
            return {
                "label": label,
                "passed": False,
                "error": f"请求异常: {e}",
            }
