"""Session 管理: 创建 / 加载 / 查看 session。"""

import json
import os
import sys
from datetime import datetime


def create_session(backend):
    """创建新 session 目录, 并记录到 .current_session。

    Returns:
        output_dir: 新建的 session 目录路径。
    """
    log_dir = backend.log_dir
    session_file = os.path.join(log_dir, ".current_session")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(log_dir, f"bench_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 让后端执行额外操作 (如 trtllm 生成配置文件)
    backend.on_session_created(output_dir)

    # 保存启动命令
    cmd_file = os.path.join(output_dir, "server_cmd.txt")
    with open(cmd_file, "w") as f:
        server_cmd = backend.build_server_cmd(output_dir)
        f.write(" \\\n    ".join(server_cmd) + "\n")

    # 写入 session 文件
    os.makedirs(os.path.dirname(session_file), exist_ok=True)
    with open(session_file, "w") as f:
        f.write(output_dir)

    print(f"[Pipeline] 新建 session: {output_dir}")
    return output_dir


def load_session(backend):
    """读取当前 session 目录。"""
    session_file = os.path.join(backend.log_dir, ".current_session")

    if not os.path.exists(session_file):
        print(f"[Pipeline] ✗ 未找到 session 文件: {session_file}")
        print(f"[Pipeline]   请先运行不带 --no-server 的命令创建 session")
        sys.exit(1)

    with open(session_file, "r") as f:
        output_dir = f.read().strip()

    if not os.path.isdir(output_dir):
        print(f"[Pipeline] ✗ session 目录不存在: {output_dir}")
        sys.exit(1)

    print(f"[Pipeline] 复用 session: {output_dir}")
    return output_dir


def show_summary(backend, session_name=None):
    """从指定或当前 session 的 summary.json 读取并打印所有轮次结果表格。"""
    from .metrics import print_round_table

    log_dir = backend.log_dir

    if session_name:
        output_dir = os.path.join(log_dir, session_name)
        if not os.path.isdir(output_dir):
            print(f"[Pipeline] ✗ 指定的 session 目录不存在: {output_dir}")
            if os.path.isdir(log_dir):
                sessions = sorted(
                    [d for d in os.listdir(log_dir)
                     if d.startswith("bench_") and os.path.isdir(os.path.join(log_dir, d))],
                    reverse=True,
                )
                if sessions:
                    print(f"[Pipeline] 可用的 session:")
                    for s in sessions:
                        print(f"             {s}")
            sys.exit(1)
        print(f"[Pipeline] 指定 session: {output_dir}")
    else:
        output_dir = load_session(backend)

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
