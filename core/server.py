"""服务管理: 启动 / 健康检查 / 终止。"""

import os
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


def start_server(backend, output_dir):
    """后台启动推理服务，日志重定向到 server.log。"""
    server_cmd = backend.build_server_cmd(output_dir)
    log_path = os.path.join(output_dir, "server.log")
    log_file = open(log_path, "w")
    print(f"[Pipeline] 启动 {backend.name} server ...")
    print(f"[Pipeline] Server 日志: {log_path}")
    proc = subprocess.Popen(
        server_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file


def wait_for_server(backend):
    """轮询 /health 端点等待服务就绪。"""
    url = f"http://127.0.0.1:{backend.port}/health"
    timeout = backend.health_check_timeout
    interval = backend.health_check_interval
    print(f"[Pipeline] 等待服务就绪 (轮询 {url}, 超时 {timeout}s) ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urlopen(url, timeout=5)
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"[Pipeline] ✓ 服务就绪 (耗时 {elapsed:.1f}s)")
                return True
        except (URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)
    print(f"[Pipeline] ✗ 服务启动超时 ({timeout}s)")
    return False


def check_server_alive(backend):
    """检查服务是否在线 (单次检查)。"""
    url = f"http://127.0.0.1:{backend.port}/health"
    try:
        resp = urlopen(url, timeout=5)
        return resp.status == 200
    except (URLError, ConnectionError, OSError):
        return False


def kill_server(backend, proc):
    """终止推理服务进程组。"""
    if proc and proc.poll() is None:
        print(f"\n[Pipeline] 正在终止 {backend.name} server (pid={proc.pid}) ...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
            print("[Pipeline] ✓ Server 已停止")
        except Exception:
            print("[Pipeline] 强制 kill server ...")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
