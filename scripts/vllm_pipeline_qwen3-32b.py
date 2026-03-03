#!/usr/bin/env python3
"""
vllm 统一压测 Pipeline (多轮, 支持 session)
=============================================
流程: 启动服务 → 健康检查 → 多轮逐级压测 → 日志收集 → 结果汇总 → 清理

用法:
    # 首次运行: 启动服务 + 压测 (完成后自动 kill 服务)
    python3 vllm_pipeline_qwen3-32b.py

    # 只启动服务, 不跑压测 (服务保持运行)
    python3 vllm_pipeline_qwen3-32b.py --server-only

    # 只跑压测, 不启动服务 (复用已运行的服务, 结果写入同一 session 目录)
    python3 vllm_pipeline_qwen3-32b.py --no-server
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from urllib.error import URLError
from urllib.request import urlopen

# ==================== 配置区 ====================

# 模型配置
MODEL_PATH = "/data/models/Qwen3-32B-FP8/checkpoints"
MODEL_NAME = "Qwen3-32B"

# 服务配置
HOST = "0.0.0.0"
PORT = 8000

# 压测配置
TOKENIZER = MODEL_PATH
SEED = 12138

# 多轮测试配置: 每轮有独立的 input_len, output_len, concurrency_list
TEST_ROUNDS = [
    {
        "label": "round1_prefill",
        "desc": "Prefill 压测 (长输入, 短输出)",
        "input_len": 5500,
        "output_len": 1,
        "concurrency_list": [1],
    },
    {
        "label": "round2_decode",
        "desc": "Decode 压测 (短输入, 长输出)",
        "input_len": 1,
        "output_len": 600,
        "concurrency_list": [64, 128, 256],
    },
    {
        "label": "round3_mixed",
        "desc": "混合压测 (长输入, 长输出)",
        "input_len": 5500,
        "output_len": 600,
        "concurrency_list": [1],
    },
]

# 健康检查配置
HEALTH_CHECK_INTERVAL = 5    # 每次检查间隔(秒)
HEALTH_CHECK_TIMEOUT = 600   # 超时时间(秒)

# Session 文件: 记录当前 session 的输出目录路径
SESSION_FILE = "/data/logs/vllm/.current_session"
LOG_BASE_DIR = "/data/logs/vllm"

# ==================== 服务启动命令 ====================

SERVER_CMD = [
    "vllm", "serve", MODEL_PATH,
    "--served-model-name", MODEL_NAME,
    "--trust-remote-code",
    "--reasoning-parser", "qwen3",
    "--host", HOST,
    "--port", str(PORT),
    "--tensor-parallel-size", "1",
    "--kv-cache-dtype", "fp8_e4m3",
    "--max-model-len", "6144",
    "--max-num-seqs", "2048",
    "--max-num-batched-tokens", "8192",
    "--enable-chunked-prefill",
    "--gpu-memory-utilization", "0.95", 
]


# ==================== Session 管理 ====================

def create_session():
    """创建新 session 目录, 并记录到 .current_session。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(LOG_BASE_DIR, f"bench_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存启动命令
    cmd_file = os.path.join(output_dir, "server_cmd.txt")
    with open(cmd_file, "w") as f:
        f.write(" \\\n    ".join(SERVER_CMD) + "\n")

    # 写入 session 文件
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        f.write(output_dir)

    print(f"[Pipeline] 新建 session: {output_dir}")
    return output_dir


def load_session():
    """读取当前 session 目录。"""
    if not os.path.exists(SESSION_FILE):
        print(f"[Pipeline] ✗ 未找到 session 文件: {SESSION_FILE}")
        print(f"[Pipeline]   请先运行不带 --no-server 的命令创建 session")
        sys.exit(1)

    with open(SESSION_FILE, "r") as f:
        output_dir = f.read().strip()

    if not os.path.isdir(output_dir):
        print(f"[Pipeline] ✗ session 目录不存在: {output_dir}")
        sys.exit(1)

    print(f"[Pipeline] 复用 session: {output_dir}")
    return output_dir


# ==================== 服务管理 ====================

def start_server(output_dir):
    """后台启动 vllm server，日志重定向到 server.log。"""
    log_path = os.path.join(output_dir, "server.log")
    log_file = open(log_path, "w")
    print(f"[Pipeline] 启动 vllm server ...")
    print(f"[Pipeline] Server 日志: {log_path}")
    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # 创建新进程组，方便后续 kill 整个组
    )
    return proc, log_file


def wait_for_server():
    """轮询 /health 端点等待服务就绪。"""
    url = f"http://127.0.0.1:{PORT}/health"
    print(f"[Pipeline] 等待服务就绪 (轮询 {url}, 超时 {HEALTH_CHECK_TIMEOUT}s) ...")
    start = time.time()
    while time.time() - start < HEALTH_CHECK_TIMEOUT:
        try:
            resp = urlopen(url, timeout=5)
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"[Pipeline] ✓ 服务就绪 (耗时 {elapsed:.1f}s)")
                return True
        except (URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    print(f"[Pipeline] ✗ 服务启动超时 ({HEALTH_CHECK_TIMEOUT}s)")
    return False


def check_server_alive():
    """检查服务是否在线 (单次检查)。"""
    url = f"http://127.0.0.1:{PORT}/health"
    try:
        resp = urlopen(url, timeout=5)
        return resp.status == 200
    except (URLError, ConnectionError, OSError):
        return False


def kill_server(proc):
    """终止 vllm server 进程组。"""
    if proc and proc.poll() is None:
        print(f"\n[Pipeline] 正在终止 vllm server (pid={proc.pid}) ...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
            print("[Pipeline] ✓ Server 已停止")
        except Exception:
            print("[Pipeline] 强制 kill server ...")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)


# ==================== 压测执行 ====================

def run_benchmark(concurrency, input_len, output_len, round_dir):
    """运行单轮压测，返回结果文件路径。"""
    num_prompts = concurrency * 5
    result_path = os.path.join(round_dir, f"bench_c{concurrency}.json")
    log_path = os.path.join(round_dir, f"bench_c{concurrency}.log")

    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--model", MODEL_NAME,
        "--tokenizer", TOKENIZER,
        "--max-concurrency", str(concurrency),
        "--num-prompts", str(num_prompts),
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--request-rate", "inf",
        "--save-result",
        "--save-detailed",
        "--result-filename", result_path,
        "--seed", str(SEED),
    ]

    print(f"\n{'='*60}")
    print(f"[压测] max-concurrency={concurrency}, num-prompts={num_prompts}")
    print(f"       input_len={input_len}, output_len={output_len}")
    print(f"{'='*60}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # 同时输出到终端和日志文件
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            log_file.write(decoded)
        proc.wait()

    if proc.returncode != 0:
        print(f"[Pipeline] ⚠ 并发={concurrency} 压测异常退出 (code={proc.returncode})")
        return None

    print(f"[Pipeline] ✓ 并发={concurrency} 完成, 结果: {result_path}")
    return result_path


# ==================== 结果处理 ====================

def extract_metrics(result_path):
    """从 vllm bench 结果 JSON 文件中提取关键指标。

    注意: vllm 的 JSON 字段名与 sglang 不同:
    - 没有 input_throughput, 需要从 total_input_tokens / duration 计算
    - total_token_throughput 对应 sglang 的 total_throughput
    - 没有 e2e_latency 相关字段
    """
    try:
        with open(result_path, "r") as f:
            data = json.load(f)

        # 计算 input_throughput (vllm 不直接提供)
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


def _v(val, default=0):
    """安全取值: None 时返回 default, 防止 format 报错。"""
    return default if val is None else val


def print_round_table(round_cfg, results):
    """打印单轮结果表格。"""
    label = round_cfg["label"]
    desc = round_cfg["desc"]
    isl = round_cfg["input_len"]
    osl = round_cfg["output_len"]

    print(f"\n{'='*130}")
    print(f"  [{label}] {desc}  (ISL={isl}, OSL={osl})")
    print(f"{'='*130}")
    header = (
        f"{'Concur':>6} | {'Completed':>9} | {'Req/s':>8} | {'InTok/s':>10} | {'OutTok/s':>10} | "
        f"{'TTFT_mean':>10} | {'TTFT_p99':>10} | "
        f"{'TPOT_mean':>10} | {'TPOT_p99':>10} | "
        f"{'ITL_mean':>10} | {'ITL_p99':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{_v(r.get('concurrency'), '-'):>6} | "
            f"{_v(r.get('completed'), '-'):>9} | "
            f"{_v(r.get('request_throughput')):>8.2f} | "
            f"{_v(r.get('input_throughput')):>10.1f} | "
            f"{_v(r.get('output_throughput')):>10.1f} | "
            f"{_v(r.get('mean_ttft_ms')):>10.2f} | "
            f"{_v(r.get('p99_ttft_ms')):>10.2f} | "
            f"{_v(r.get('mean_tpot_ms')):>10.2f} | "
            f"{_v(r.get('p99_tpot_ms')):>10.2f} | "
            f"{_v(r.get('mean_itl_ms')):>10.2f} | "
            f"{_v(r.get('p99_itl_ms')):>10.2f}"
        )
    print(f"{'='*130}")


def generate_summary(all_round_results, output_dir):
    """生成汇总 summary.json 并打印所有轮次结果。

    如果 summary.json 已存在 (多次 --no-server 运行), 则合并历史结果。
    """
    summary_path = os.path.join(output_dir, "summary.json")

    # 加载已有的 summary (如果存在)
    existing_rounds = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing = json.load(f)
            existing_rounds = existing.get("rounds", [])
        except Exception:
            pass

    # 构建本次结果
    new_rounds = []
    for round_cfg, results in all_round_results:
        round_entry = {
            "label": round_cfg["label"],
            "desc": round_cfg["desc"],
            "input_len": round_cfg["input_len"],
            "output_len": round_cfg["output_len"],
            "concurrency_list": round_cfg["concurrency_list"],
            "results": results,
        }
        new_rounds.append(round_entry)
        print_round_table(round_cfg, results)

    # 合并: 同 label 的用新结果覆盖, 不同的保留
    existing_labels = {r["label"] for r in new_rounds}
    merged_rounds = [r for r in existing_rounds if r["label"] not in existing_labels]
    merged_rounds.extend(new_rounds)

    summary = {
        "server_cmd": " ".join(SERVER_CMD),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "seed": SEED,
        "rounds": merged_rounds,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Pipeline] 汇总文件: {summary_path}")


# ==================== 主流程 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="vllm 统一压测 Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--server-only",
        action="store_true",
        help="只启动服务, 不跑压测 (服务保持运行, 后续用 --no-server 跑压测)",
    )
    group.add_argument(
        "--no-server",
        action="store_true",
        help="只跑压测, 不启动服务 (复用已运行的服务和 session 目录)",
    )
    group.add_argument(
        "--show-summary",
        nargs="?",
        const="current_session",
        help="只查看已有的压测结果。可以指定具体的 session 目录 (如: logs/vllm/bench_...), 不指定则默认查看当前 session",
    )
    return parser.parse_args()


def show_summary(session_dir=None):
    """从指定的 session 目录或当前 session 的 summary.json 读取并打印所有轮次结果表格。"""
    if session_dir and session_dir != "current_session":
        output_dir = session_dir
    else:
        output_dir = load_session()
    
    summary_path = os.path.join(output_dir, "summary.json")

    if not os.path.exists(summary_path):
        print(f"[Pipeline] ✗ 未找到汇总文件: {summary_path}")
        print(f"[Pipeline]   请先跑完压测生成结果")
        sys.exit(1)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    rounds = summary.get("rounds", [])
    if not rounds:
        print("[Pipeline] ⚠ summary.json 中没有压测结果")
        sys.exit(1)

    print(f"[Pipeline] Session: {output_dir}")
    print(f"[Pipeline] 模型: {summary.get('model', '-')}")
    print(f"[Pipeline] 时间: {summary.get('timestamp', '-')}")

    for rnd in rounds:
        round_cfg = {
            "label": rnd["label"],
            "desc": rnd["desc"],
            "input_len": rnd["input_len"],
            "output_len": rnd["output_len"],
        }
        print_round_table(round_cfg, rnd.get("results", []))


def main():
    args = parse_args()

    # ---- --show-summary: 只看结果, 立即返回 ----
    if args.show_summary:
        show_summary(args.show_summary)
        return

    server_proc = None
    server_log = None

    # 确定 session 目录
    if args.no_server:
        output_dir = load_session()
    else:
        output_dir = create_session()

    try:
        # ---- 服务管理 ----
        if args.no_server:
            # --no-server 模式: 检查服务是否在线
            if not check_server_alive():
                print("[Pipeline] ✗ 服务未运行, 请先启动服务")
                sys.exit(1)
            print("[Pipeline] ✓ 服务已在线")
        else:
            # 启动服务
            server_proc, server_log = start_server(output_dir)
            if not wait_for_server():
                print("[Pipeline] 服务启动失败，退出")
                sys.exit(1)

        # ---- --server-only: 启动后挂起等待 ----
        if args.server_only:
            print(f"\n[Pipeline] 服务已启动, --server-only 模式")
            print(f"[Pipeline] 压测时请运行: python3 {__file__} --no-server")
            print(f"[Pipeline] 按 Ctrl+C 停止服务")
            try:
                server_proc.wait()
            except KeyboardInterrupt:
                print("\n[Pipeline] 收到 Ctrl+C, 正在停止服务 ...")
            return

        # ---- 多轮压测 ----
        all_round_results = []
        for i, round_cfg in enumerate(TEST_ROUNDS, 1):
            label = round_cfg["label"]
            desc = round_cfg["desc"]
            input_len = round_cfg["input_len"]
            output_len = round_cfg["output_len"]
            concurrency_list = round_cfg["concurrency_list"]

            print(f"\n{'#'*70}")
            print(f"  第 {i}/{len(TEST_ROUNDS)} 轮: [{label}] {desc}")
            print(f"  ISL={input_len}, OSL={output_len}, 并发={concurrency_list}")
            print(f"{'#'*70}")

            # 为每轮创建子目录
            round_dir = os.path.join(output_dir, label)
            os.makedirs(round_dir, exist_ok=True)

            round_results = []
            for c in concurrency_list:
                result_path = run_benchmark(c, input_len, output_len, round_dir)
                if result_path:
                    metrics = extract_metrics(result_path)
                    if metrics:
                        round_results.append(metrics)

            all_round_results.append((round_cfg, round_results))

        # ---- 生成汇总 ----
        has_results = any(results for _, results in all_round_results)
        if has_results:
            generate_summary(all_round_results, output_dir)
        else:
            print("[Pipeline] ⚠ 没有有效的压测结果")

    finally:
        # 清理: server_proc 为 None 时 (--no-server) 不会执行 kill
        if server_proc:
            kill_server(server_proc)
        if server_log:
            server_log.close()
        print(f"\n[Pipeline] 完成! 所有输出在: {output_dir}")


if __name__ == "__main__":
    main()
