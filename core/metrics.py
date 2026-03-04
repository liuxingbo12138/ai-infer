"""结果处理: 表格打印 / 汇总生成。"""

import json
import os
from datetime import datetime


def _v(val, default=0):
    """安全取值: None 时返回 default, 防止 format 报错。"""
    return default if val is None else val


def print_round_table(round_cfg, results):
    """打印单轮结果表格。"""
    label = round_cfg["label"]
    desc = round_cfg["desc"]
    isl = round_cfg["input_len"]
    osl = round_cfg["output_len"]

    print(f"\n{'='*140}")
    print(f"  [{label}] {desc}  (ISL={isl}, OSL={osl})")
    print(f"{'='*140}")
    header = (
        f"{'Concur':>6} | {'Completed':>9} | {'Req/s':>8} | {'InTok/s':>10} | {'OutTok/s':>10} | "
        f"{'TTFT_mean':>10} | {'TTFT_p99':>10} | "
        f"{'TPOT_mean':>10} | {'TPOT_p99':>10} | "
        f"{'ITL_mean':>10} | {'ITL_p99':>10} | "
        f"{'E2E_mean':>10} | {'E2E_p99':>10}"
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
            f"{_v(r.get('p99_itl_ms')):>10.2f} | "
            f"{_v(r.get('mean_e2e_latency_ms')):>10.2f} | "
            f"{_v(r.get('p99_e2e_latency_ms')):>10.2f}"
        )
    print(f"{'='*140}")


def generate_summary(backend, all_round_results, output_dir):
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
        "server_cmd": backend.server_cmd_for_summary(output_dir),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "backend": backend.name,
        "model": backend.model_name,
        "model_path": backend.model_path,
        "seed": backend.seed,
        "rounds": merged_rounds,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Pipeline] 汇总文件: {summary_path}")
