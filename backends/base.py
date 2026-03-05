"""后端抽象基类: 每个推理框架只需实现 3 个方法。"""

import os
from abc import ABC, abstractmethod


class BackendBase(ABC):
    """推理框架后端抽象基类。"""

    def __init__(self, config: dict):
        self.config = config

    # ---------- 子类必须实现 ----------

    @abstractmethod
    def build_server_cmd(self, output_dir: str) -> list[str]:
        """构建服务启动命令列表。

        Args:
            output_dir: session 输出目录 (trtllm 需要在此生成配置文件)。

        Returns:
            可直接传给 subprocess.Popen 的命令列表。
        """

    @abstractmethod
    def build_bench_cmd(
        self,
        concurrency: int,
        input_len: int,
        output_len: int,
        num_prompts: int,
        result_path: str,
    ) -> list[str]:
        """构建压测命令列表。

        Args:
            concurrency: 最大并发数。
            input_len: 随机输入 token 长度。
            output_len: 随机输出 token 长度。
            num_prompts: 请求总数。
            result_path: 结果输出文件路径。

        Returns:
            可直接传给 subprocess.Popen 的命令列表。
        """

    @abstractmethod
    def parse_result(self, result_path: str) -> dict | None:
        """解析压测结果文件, 返回统一指标字典。

        Returns:
            统一格式的指标字典, 解析失败返回 None。
            字段: concurrency, completed, request_throughput, input_throughput,
                  output_throughput, total_throughput, mean_ttft_ms, median_ttft_ms,
                  p99_ttft_ms, mean_tpot_ms, median_tpot_ms, p99_tpot_ms,
                  mean_itl_ms, median_itl_ms, p99_itl_ms, mean_e2e_latency_ms,
                  median_e2e_latency_ms, p99_e2e_latency_ms, duration
        """

    # ---------- 公共辅助 ----------

    @property
    def name(self) -> str:
        return self.config["backend"]

    @property
    def model_name(self) -> str:
        return self.config["model"]["name"]

    @property
    def model_path(self) -> str:
        return self.config["model"]["path"]

    @property
    def host(self) -> str:
        return self.config.get("server", {}).get("host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self.config.get("server", {}).get("port", 8000)

    @property
    def seed(self) -> int:
        return self.config.get("benchmark", {}).get("seed", 12138)

    @property
    def tokenizer(self) -> str:
        return self.config.get("benchmark", {}).get("tokenizer", self.model_path)

    @property
    def test_rounds(self) -> list[dict]:
        return self.config.get("benchmark", {}).get("rounds", [])

    @property
    def log_dir(self) -> str:
        """日志根目录, 自动追加 model_name 子目录。

        如果 YAML 中为相对路径, 则相对于项目根目录解析。
        最终结构: {log_dir}/{model_name}/
        """
        d = self.config.get("log_dir", f"logs/{self.name}")
        if not os.path.isabs(d):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            d = os.path.join(project_root, d)
        return os.path.join(d, self.model_name)

    @property
    def health_check_interval(self) -> int:
        return self.config.get("server", {}).get("health_check_interval", 5)

    @property
    def health_check_timeout(self) -> int:
        return self.config.get("server", {}).get("health_check_timeout", 600)

    def result_extension(self) -> str:
        """结果文件扩展名 (sglang 用 .jsonl, 其他用 .json)。"""
        return ".json"

    def on_session_created(self, output_dir: str) -> None:
        """Session 创建后的额外操作 (例如 trtllm 需要生成 config yaml)。

        默认不做任何事。子类可覆盖。
        """

    def server_cmd_for_summary(self, output_dir: str) -> str:
        """用于 summary.json 中记录的服务启动命令字符串。"""
        return " ".join(self.build_server_cmd(output_dir))
