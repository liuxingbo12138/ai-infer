#!/usr/bin/env python3
"""
统一压测 Pipeline (多轮, 支持 session, 配置驱动)
==================================================
流程: 加载配置 → 启动服务 → 健康检查 → 多轮逐级压测 → 日志收集 → 结果汇总 → 清理

用法:
    # 首次运行: 启动服务 + 压测 (完成后自动 kill 服务)
    python3 bench.py -c configs/sglang_qwen3-32b.yaml

    # 只启动服务, 不跑压测 (服务保持运行)
    python3 bench.py -c configs/sglang_qwen3-32b.yaml --server-only

    # 只跑压测, 不启动服务 (复用已运行的服务, 结果写入同一 session 目录)
    python3 bench.py -c configs/sglang_qwen3-32b.yaml --no-server

    # 只看结果 (从当前 session 的 summary.json 读取并打印表格)
    python3 bench.py -c configs/sglang_qwen3-32b.yaml --show-summary

    # 查看指定 session 的结果
    python3 bench.py -c configs/sglang_qwen3-32b.yaml --show-summary bench_20260302_092317
"""

import argparse
import os
import sys
import yaml

from backends import get_backend
from core.session import create_session, load_session, show_summary
from core.server import start_server, wait_for_server, check_server_alive, kill_server
from tests import get_test, ALL_TEST_NAMES
from core.benchmark import run_benchmark
from core.metrics import generate_summary


def parse_args():
    parser = argparse.ArgumentParser(description="统一推理压测 Pipeline")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="YAML 配置文件路径 (如 configs/sglang_qwen3-32b.yaml)",
    )
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
        const=True,
        default=None,
        metavar="SESSION_NAME",
        help="只查看已有的压测结果, 可指定 session 目录名 (如 bench_20260302_092317)",
    )
    parser.add_argument(
        "--test",
        choices=ALL_TEST_NAMES,
        default=None,
        metavar="TEST_TYPE",
        help=f"运行功能测试 (可选: {', '.join(ALL_TEST_NAMES)}), 不执行性能压测",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="功能测试的样本数量 (默认: tool_calling/structured_output=5, gsm8k=100, mmlu=50)",
    )
    return parser.parse_args()


def load_config(config_path):
    """加载 YAML 配置文件。"""
    if not os.path.exists(config_path):
        print(f"[Pipeline] ✗ 配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 实例化后端
    backend_cls = get_backend(config["backend"])
    backend = backend_cls(config)

    print(f"[Pipeline] 后端: {backend.name}")
    print(f"[Pipeline] 模型: {backend.model_name}")
    print(f"[Pipeline] 配置: {args.config}")

    # ---- --show-summary: 只看结果, 立即返回 ----
    if args.show_summary is not None:
        session_name = args.show_summary if args.show_summary is not True else None
        show_summary(backend, session_name)
        return

    server_proc = None
    server_log = None

    # 确定 session 目录
    if args.no_server:
        output_dir = load_session(backend)
    else:
        output_dir = create_session(backend)

    try:
        # ---- 服务管理 ----
        if args.no_server:
            if not check_server_alive(backend):
                print("[Pipeline] ✗ 服务未运行, 请先启动服务")
                sys.exit(1)
            print("[Pipeline] ✓ 服务已在线")
        else:
            server_proc, server_log = start_server(backend, output_dir)
            if not wait_for_server(backend):
                print("[Pipeline] 服务启动失败，退出")
                sys.exit(1)

        # ---- --server-only: 启动后挂起等待 ----
        if args.server_only:
            print(f"\n[Pipeline] 服务已启动, --server-only 模式")
            print(f"[Pipeline] 压测时请运行: python3 bench.py -c {args.config} --no-server")
            print(f"[Pipeline] 功能测试: python3 bench.py -c {args.config} --no-server --test <类型>")
            print(f"[Pipeline] 按 Ctrl+C 停止服务")
            try:
                server_proc.wait()
            except KeyboardInterrupt:
                print("\n[Pipeline] 收到 Ctrl+C, 正在停止服务 ...")
            return

        # ---- --test: 功能测试模式 ----
        if args.test:
            test_cls = get_test(args.test)
            test = test_cls(backend, num_samples=args.num_samples)
            test.execute(output_dir=output_dir)
            return

        # ---- 多轮压测 ----
        test_rounds = backend.test_rounds
        all_round_results = []
        for i, round_cfg in enumerate(test_rounds, 1):
            label = round_cfg["label"]
            desc = round_cfg["desc"]
            input_len = round_cfg["input_len"]
            output_len = round_cfg["output_len"]
            concurrency_list = round_cfg["concurrency_list"]

            print(f"\n{'#'*70}")
            print(f"  第 {i}/{len(test_rounds)} 轮: [{label}] {desc}")
            print(f"  ISL={input_len}, OSL={output_len}, 并发={concurrency_list}")
            print(f"{'#'*70}")

            if not concurrency_list:
                print(f"[Pipeline] ⚠ [{label}] concurrency_list 为空, 跳过本轮")
                continue

            round_dir = os.path.join(output_dir, label)
            os.makedirs(round_dir, exist_ok=True)

            round_results = []
            for c in concurrency_list:
                result_path = run_benchmark(backend, c, input_len, output_len, round_dir)
                if result_path:
                    metrics = backend.parse_result(result_path)
                    if metrics:
                        round_results.append(metrics)

            all_round_results.append((round_cfg, round_results))

        # ---- 生成汇总 ----
        has_results = any(results for _, results in all_round_results)
        if has_results:
            generate_summary(backend, all_round_results, output_dir)
        else:
            print("[Pipeline] ⚠ 没有有效的压测结果")

    finally:
        if server_proc:
            kill_server(backend, server_proc)
        if server_log:
            server_log.close()
        print(f"\n[Pipeline] 完成! 所有输出在: {output_dir}")


if __name__ == "__main__":
    main()
