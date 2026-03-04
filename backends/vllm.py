"""vLLM 后端实现。"""

import json

from .base import BackendBase


class VLLMBackend(BackendBase):

    def build_server_cmd(self, output_dir: str) -> list[str]:
        cmd = [
            "vllm", "serve", self.model_path,
            "--served-model-name", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
        ]
        for arg in self.config.get("server", {}).get("extra_args", []):
            cmd.extend(arg.split("=", 1) if "=" in arg else [arg])
        return cmd

    def build_bench_cmd(self, concurrency, input_len, output_len, num_prompts, result_path):
        return [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--base-url", f"http://127.0.0.1:{self.port}",
            "--model", self.model_name,
            "--tokenizer", self.tokenizer,
            "--max-concurrency", str(concurrency),
            "--num-prompts", str(num_prompts),
            "--dataset-name", "random",
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--request-rate", "inf",
            "--save-result",
            "--save-detailed",
            "--result-filename", result_path,
            "--seed", str(self.seed),
        ]

    def parse_result(self, result_path):
        try:
            with open(result_path, "r") as f:
                data = json.load(f)

            duration = data.get("duration", 0)
            total_input = data.get("total_input_tokens", 0)
            input_throughput = (total_input / duration) if duration else 0

            return {
                "concurrency": data.get("max_concurrency"),
                "completed": data.get("completed"),
                "request_throughput": data.get("request_throughput"),
                "input_throughput": input_throughput,
                "output_throughput": data.get("output_throughput"),
                "total_throughput": data.get("total_token_throughput"),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "median_ttft_ms": data.get("median_ttft_ms"),
                "p99_ttft_ms": data.get("p99_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "median_tpot_ms": data.get("median_tpot_ms"),
                "p99_tpot_ms": data.get("p99_tpot_ms"),
                "mean_itl_ms": data.get("mean_itl_ms"),
                "median_itl_ms": data.get("median_itl_ms"),
                "p99_itl_ms": data.get("p99_itl_ms"),
                "duration": duration,
            }
        except Exception as e:
            print(f"[Pipeline] ⚠ 解析 {result_path} 失败: {e}")
            return None
