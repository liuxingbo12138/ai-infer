"""压测执行: 运行单轮压测。"""

import os
import subprocess
import sys


def run_benchmark(backend, concurrency, input_len, output_len, round_dir):
    """运行单轮压测，返回结果文件路径。"""
    num_prompts = concurrency * 5
    ext = backend.result_extension()
    result_path = os.path.join(round_dir, f"bench_c{concurrency}{ext}")
    log_path = os.path.join(round_dir, f"bench_c{concurrency}.log")

    cmd = backend.build_bench_cmd(
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        num_prompts=num_prompts,
        result_path=result_path,
    )

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
