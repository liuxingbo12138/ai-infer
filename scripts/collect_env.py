#!/usr/bin/env python3
"""
Universal system environment information collector for AI inference projects.

Collects comprehensive system, hardware, and software environment information
useful for debugging, bug reporting, and environment verification.

References:
  - vllm/vllm/collect_env.py
  - sglang/python/sglang/check_env.py

Usage:
  python scripts/collect_env.py           # Human-readable output
  python scripts/collect_env.py --json    # JSON output
"""

import argparse
import importlib.metadata
import json
import locale
import os
import platform
import subprocess
import sys
from collections import OrderedDict

# ============================================================================
#  Core utility: safe subprocess execution
# ============================================================================

def run_cmd(command, shell=False, timeout=10):
    """
    Safely run a command and return (return_code, stdout, stderr).
    Never raises on missing commands or timeouts.
    """
    if isinstance(command, str):
        shell = True
    try:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell,
        )
        raw_out, raw_err = p.communicate(timeout=timeout)
        enc = "oem" if sys.platform.startswith("win") else locale.getpreferredencoding()
        return p.returncode, raw_out.decode(enc, errors="replace").strip(), raw_err.decode(enc, errors="replace").strip()
    except FileNotFoundError:
        cmd_str = command if isinstance(command, str) else command[0]
        return 127, "", f"Command not found: {cmd_str}"
    except subprocess.TimeoutExpired:
        p.kill()
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def run_and_read(command):
    """Run command, return stdout if success, else None."""
    rc, out, _ = run_cmd(command)
    return out if rc == 0 and out else None


def run_and_match(command, regex):
    """Run command and return the first regex group match, or None."""
    import re
    rc, out, _ = run_cmd(command)
    if rc != 0 or not out:
        return None
    m = re.search(regex, out)
    return m.group(1) if m else None


# ============================================================================
#  Platform detection
# ============================================================================

def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    elif sys.platform.startswith("win"):
        return "win32"
    return sys.platform


# ============================================================================
#  1. System / OS info
# ============================================================================

def get_os_info():
    info = OrderedDict()
    info["Hostname"] = platform.node()
    info["Platform"] = platform.platform()
    info["Architecture"] = platform.machine()

    plat = get_platform()
    if plat == "linux":
        desc = run_and_match("lsb_release -a", r"Description:\t(.*)")
        if desc is None:
            desc = run_and_match("cat /etc/*-release", r'PRETTY_NAME="(.*)"')
        info["OS"] = f"{desc} ({platform.machine()})" if desc else platform.platform()
        info["Kernel"] = platform.release()
        libc = platform.libc_ver()
        info["Libc"] = f"{libc[0]}-{libc[1]}" if libc[0] else "N/A"
    elif plat == "darwin":
        ver = run_and_read("sw_vers -productVersion")
        info["OS"] = f"macOS {ver} ({platform.machine()})" if ver else platform.platform()
        info["Kernel"] = platform.release()
        info["Libc"] = "N/A"
    elif plat == "win32":
        info["OS"] = platform.platform()
        info["Kernel"] = platform.version()
        info["Libc"] = "N/A"

    return info


# ============================================================================
#  2. Python environment
# ============================================================================

def get_python_info():
    info = OrderedDict()
    info["Python Version"] = sys.version.replace("\n", " ")
    info["Python Path"] = sys.executable
    info["Python Platform"] = platform.python_implementation()

    # Virtual environment detection
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        info["Virtual Env"] = venv
    elif hasattr(sys, "real_prefix"):
        info["Virtual Env"] = sys.prefix  # old-style virtualenv
    elif sys.prefix != sys.base_prefix:
        info["Virtual Env"] = sys.prefix
    else:
        info["Virtual Env"] = "None detected"

    # Conda detection
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    if conda:
        info["Conda Env"] = conda

    return info


# ============================================================================
#  3. Toolchain
# ============================================================================

def get_toolchain_info():
    info = OrderedDict()
    info["GCC"] = run_and_match("gcc --version", r"gcc.*?(\d+\.\d+\.\d+)") or "Not found"
    info["Clang"] = run_and_match("clang --version", r"clang version (\S+)") or "Not found"
    info["CMake"] = run_and_match("cmake --version", r"cmake version (\S+)") or "Not found"
    return info


# ============================================================================
#  4. CPU info
# ============================================================================

def get_cpu_info():
    info = OrderedDict()
    plat = get_platform()

    if plat == "linux":
        # Parse lscpu for key fields
        output = run_and_read("lscpu")
        if output:
            fields = {
                "Model name": "CPU Model",
                "CPU(s)": "CPU Count",
                "Socket(s)": "Sockets",
                "Core(s) per socket": "Cores per Socket",
                "Thread(s) per core": "Threads per Core",
                "NUMA node(s)": "NUMA Nodes",
                "Hypervisor vendor": "Hypervisor",
            }
            for line in output.split("\n"):
                for key, label in fields.items():
                    if line.strip().startswith(key):
                        val = line.split(":", 1)[1].strip()
                        info[label] = val
        else:
            info["CPU Model"] = platform.processor() or "Unknown"
    elif plat == "darwin":
        brand = run_and_read("sysctl -n machdep.cpu.brand_string")
        info["CPU Model"] = brand or platform.processor() or "Unknown"
        cores = run_and_read("sysctl -n hw.ncpu")
        info["CPU Count"] = cores or "Unknown"
        pcores = run_and_read("sysctl -n hw.physicalcpu")
        if pcores:
            info["Physical Cores"] = pcores
    elif plat == "win32":
        info["CPU Model"] = platform.processor() or "Unknown"
        info["CPU Count"] = str(os.cpu_count() or "Unknown")

    return info


# ============================================================================
#  5. NVIDIA GPU info
# ============================================================================

def _get_nvidia_smi():
    """Return the nvidia-smi command path."""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        pf = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        candidates = [
            os.path.join(system_root, "System32", "nvidia-smi"),
            os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvidia-smi"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return f'"{c}"'
    return "nvidia-smi"


def get_nvidia_gpu_info():
    info = OrderedDict()

    smi = _get_nvidia_smi()
    rc, out, _ = run_cmd(f"{smi} --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
    if rc != 0:
        info["NVIDIA GPU"] = "Not detected"
        return info

    # Parse GPU list
    gpus = []
    driver_versions = set()
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append(f"{parts[0]} ({parts[2]} MiB)")
            driver_versions.add(parts[1])
        elif len(parts) >= 1:
            gpus.append(parts[0])

    for i, gpu in enumerate(gpus):
        info[f"GPU {i}"] = gpu

    if len(driver_versions) == 1:
        info["NVIDIA Driver"] = driver_versions.pop()
    elif driver_versions:
        info["NVIDIA Driver"] = ", ".join(sorted(driver_versions))

    # CUDA version from nvcc
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if os.path.isdir(cuda_home) else "nvcc"
    cuda_ver = run_and_match(f'"{nvcc}" -V', r"release .+, V([\d.]+)")
    info["CUDA (nvcc)"] = cuda_ver or "Not found"

    # CUDA driver version from nvidia-smi
    smi_cuda = run_and_match(f"{smi}", r"CUDA Version:\s+([\d.]+)")
    if smi_cuda:
        info["CUDA (Driver)"] = smi_cuda

    # cuDNN
    if get_platform() == "linux":
        rc2, cudnn_out, _ = run_cmd('ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev')
        if rc2 == 0 and cudnn_out:
            files = sorted(set(os.path.realpath(f) for f in cudnn_out.split("\n") if os.path.isfile(f)))
            info["cuDNN"] = ", ".join(files) if files else "Not found"
        else:
            info["cuDNN"] = "Not found"

    # NCCL
    nccl_ver = run_and_match('ldconfig -p | grep "libnccl.so"', r"libnccl\.so\.([\d.]+)")
    if nccl_ver:
        info["NCCL"] = nccl_ver

    # GPU Topology
    topo = run_and_read(f"{smi} topo -m")
    if topo:
        info["GPU Topology"] = "\n" + topo

    return info


# ============================================================================
#  6. AMD ROCm / HIP info
# ============================================================================

def get_rocm_gpu_info():
    info = OrderedDict()

    rc, out, _ = run_cmd("rocm-smi --showproductname --csv")
    if rc != 0:
        # ROCm not available
        return info

    info["ROCm GPU"] = out or "Detected"

    hip_ver = run_and_match("hipcc --version", r"HIP version:\s+(\S+)")
    if hip_ver:
        info["HIP Version"] = hip_ver

    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    if os.path.isdir(rocm_home):
        info["ROCM_HOME"] = rocm_home

    # ROCm driver version
    rc2, drv_out, _ = run_cmd("rocm-smi --showdriverversion --csv")
    if rc2 == 0 and drv_out:
        for line in drv_out.split("\n"):
            if "Driver version" in line or "driver" in line.lower():
                parts = line.replace('"', "").split(",")
                if len(parts) >= 2:
                    info["ROCm Driver"] = parts[-1].strip()
                    break

    # Topology
    topo = run_and_read("rocm-smi --showtopo")
    if topo:
        info["ROCm Topology"] = "\n" + topo

    return info


# ============================================================================
#  7. Ascend NPU info
# ============================================================================

def get_npu_info():
    info = OrderedDict()

    rc, out, _ = run_cmd("npu-smi info")
    if rc != 0:
        return info

    info["NPU"] = "Detected"

    # CANN version
    cann_home = None
    for var in ["ASCEND_TOOLKIT_HOME", "ASCEND_INSTALL_PATH"]:
        path = os.environ.get(var)
        if path and os.path.exists(path):
            cann_home = path
            break
    if not cann_home:
        default = "/usr/local/Ascend/ascend-toolkit/latest"
        if os.path.exists(default):
            cann_home = default

    if cann_home:
        info["CANN_HOME"] = cann_home
        ver_file = os.path.join(cann_home, "version.cfg")
        if os.path.exists(ver_file):
            try:
                with open(ver_file, "r", encoding="utf-8") as f:
                    f.readline()  # skip comment
                    ver_line = f.readline()
                    if "[" in ver_line and "]" in ver_line:
                        info["CANN Version"] = ver_line.split("[")[1].split("]")[0]
            except Exception:
                pass

    # Driver version
    rc3, board_out, _ = run_cmd("npu-smi info -t board -i 0")
    if rc3 == 0 and board_out:
        for line in board_out.split("\n"):
            if "Software Version" in line:
                info["Ascend Driver"] = line.split(":")[-1].strip()
                break

    # Topology
    rc4, topo_out, _ = run_cmd("npu-smi info -t topo")
    if rc4 == 0 and topo_out:
        info["NPU Topology"] = "\n" + topo_out

    return info


# ============================================================================
#  8. PyTorch info (optional, graceful fallback)
# ============================================================================

def get_pytorch_info():
    info = OrderedDict()
    try:
        import torch
        info["PyTorch Version"] = torch.__version__
        info["PyTorch Debug Build"] = str(torch.version.debug)

        if torch.version.cuda:
            info["PyTorch CUDA"] = torch.version.cuda
        if hasattr(torch.version, "hip") and torch.version.hip:
            info["PyTorch HIP"] = torch.version.hip

        if torch.cuda.is_available():
            info["CUDA Available"] = "True"
            info["CUDA Device Count"] = str(torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                info[f"CUDA Device {i}"] = f"{name} (Compute {cap[0]}.{cap[1]})"
        else:
            info["CUDA Available"] = "False"

    except ImportError:
        info["PyTorch"] = "Not installed"
    except Exception as e:
        info["PyTorch"] = f"Error: {e}"

    return info


# ============================================================================
#  9. Python package versions
# ============================================================================

# Packages commonly relevant to AI inference
INFERENCE_PACKAGES = [
    # Frameworks
    "torch", "torchvision", "torchaudio",
    "tensorflow", "onnxruntime", "tensorrt",
    # Inference engines
    "vllm", "sglang", "triton",
    # Kernels & acceleration
    "sgl_kernel", "flashinfer_python", "flash_attn",
    "xformers", "torchao", "xgrammar",
    # Model & data
    "transformers", "tokenizers", "safetensors",
    "huggingface_hub", "modelscope", "datasets",
    # Serving & API
    "fastapi", "uvicorn", "uvloop", "aiohttp",
    "openai", "anthropic", "litellm",
    # Infrastructure
    "numpy", "scipy", "pandas",
    "pydantic", "psutil", "packaging",
    "ray", "deepspeed",
    # Networking
    "zmq", "pyzmq", "grpcio",
    "nccl",
    # Tokenization
    "tiktoken", "sentencepiece",
]


def get_package_versions():
    info = OrderedDict()

    for pkg in INFERENCE_PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
            info[pkg] = ver
        except importlib.metadata.PackageNotFoundError:
            pass  # Skip not-installed packages silently

    return info


def get_pip_conda_packages():
    """Get filtered pip/conda package lists for relevant packages."""
    info = OrderedDict()

    patterns = {
        "torch", "numpy", "triton", "nccl", "nvidia",
        "transformers", "vllm", "sglang", "flash", "xformers",
        "tensorrt", "onnx", "deepspeed", "ray",
    }

    # pip list
    rc, out, _ = run_cmd([sys.executable, "-m", "pip", "list", "--format=freeze"])
    if rc == 0 and out:
        matched = [l for l in out.splitlines() if any(p in l.lower() for p in patterns)]
        if matched:
            info["pip packages"] = "\n" + "\n".join(f"  [pip] {l}" for l in matched)

    # conda list
    conda = os.environ.get("CONDA_EXE", "conda")
    rc2, out2, _ = run_cmd([conda, "list"])
    if rc2 == 0 and out2:
        matched2 = [
            l for l in out2.splitlines()
            if not l.startswith("#") and any(p in l.lower() for p in patterns)
        ]
        if matched2:
            info["conda packages"] = "\n" + "\n".join(f"  [conda] {l}" for l in matched2)

    return info


# ============================================================================
# 10. Environment variables
# ============================================================================

# Prefixes of env vars known to be relevant
ENV_PREFIXES = (
    "CUDA", "CUBLAS", "CUDNN", "NCCL",
    "TORCH", "PYTORCH",
    "NVIDIA", "GPU",
    "ROCM", "HIP", "HSA",
    "ASCEND", "CANN",
    "OMP_", "MKL_", "KMP_",
    "GLOO_", "TP_", "PP_",
    "VLLM", "SGLANG",
    "HF_", "HUGGING",
    "RAY_",
)

SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "auth")


def get_env_vars():
    info = OrderedDict()
    for k in sorted(os.environ.keys()):
        # Skip secrets
        if any(t in k.lower() for t in SECRET_TERMS):
            continue
        if any(k.startswith(p) for p in ENV_PREFIXES):
            info[k] = os.environ[k]
    return info


# ============================================================================
# 11. Resource limits
# ============================================================================

def get_resource_limits():
    info = OrderedDict()
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        info["ulimit -n (soft)"] = str(soft)
        info["ulimit -n (hard)"] = str(hard)
    except (ImportError, AttributeError):
        # Windows doesn't have resource module
        pass
    return info


# ============================================================================
#  Collector & formatter
# ============================================================================

SECTION_SEPARATOR = "=" * 60


def collect_all():
    """Collect all environment info into an ordered dict of sections."""
    sections = OrderedDict()
    sections["System Information"] = get_os_info()
    sections["Python Environment"] = get_python_info()
    sections["Compiler Toolchain"] = get_toolchain_info()
    sections["CPU Information"] = get_cpu_info()
    sections["NVIDIA GPU"] = get_nvidia_gpu_info()

    rocm = get_rocm_gpu_info()
    if rocm:
        sections["AMD ROCm / HIP"] = rocm

    npu = get_npu_info()
    if npu:
        sections["Ascend NPU"] = npu

    sections["PyTorch"] = get_pytorch_info()
    sections["Installed Packages (AI Inference)"] = get_package_versions()
    sections["Pip / Conda Package Details"] = get_pip_conda_packages()

    env_vars = get_env_vars()
    if env_vars:
        sections["Relevant Environment Variables"] = env_vars

    limits = get_resource_limits()
    if limits:
        sections["Resource Limits"] = limits

    return sections


def format_text(sections):
    """Format sections into human-readable text."""
    lines = []
    for title, items in sections.items():
        lines.append("")
        lines.append(SECTION_SEPARATOR)
        lines.append(f"  {title}")
        lines.append(SECTION_SEPARATOR)
        if not items:
            lines.append("  (no information collected)")
            continue
        max_key_len = max(len(k) for k in items)
        for k, v in items.items():
            v_str = str(v)
            if "\n" in v_str:
                # Multi-line values (e.g., topology, package lists)
                lines.append(f"  {k}:")
                for sub in v_str.strip().split("\n"):
                    lines.append(f"    {sub}")
            else:
                lines.append(f"  {k:<{max_key_len}} : {v_str}")
    lines.append("")
    return "\n".join(lines)


def format_json(sections):
    """Format sections into JSON."""
    return json.dumps(sections, indent=2, ensure_ascii=False)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect system environment information for AI inference debugging.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output in JSON format",
    )
    args = parser.parse_args()

    print("Collecting environment information...\n")
    sections = collect_all()

    if args.json:
        print(format_json(sections))
    else:
        print(format_text(sections))


if __name__ == "__main__":
    main()
