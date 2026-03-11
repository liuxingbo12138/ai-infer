# AI Inference Benchmark Pipeline

统一压测 Pipeline — 配置驱动 + 插件化后端架构。

本项目设计为直接运行在对应推理引擎（vLLM、SGLang、TRT-LLM）的官方 Docker 环境中。压测前，需要先启动相应容器并将代码及数据路径映射到容器内。

## 1. 启动测试容器

根据需要压测的引擎，使用以下常用的启动命令（已包含 `--gpus`、共享内存及代码数据等必要路径挂载：`-v /data:/data`）：

### TRT-LLM
```bash
docker run --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=64g --name trt-llm1 --ulimit memlock=-1 -dit --network=host --cap-add=IPC_LOCK --privileged --device=/dev/infiniband -v /nfs/gen_media/api.liandanxia:/nfs/gen_media/api.liandanxia -v /data:/data nvcr.io/nvidia/tensorrt-llm
```

### vLLM
```bash
docker run --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=64g --name vllm1 --ulimit memlock=-1 -dit --network=host --cap-add=IPC_LOCK --privileged --device=/dev/infiniband -v /nfs/gen_media/api.liandanxia:/nfs/gen_media/api.liandanxia -v /data:/data --entrypoint /bin/bash vllm/vllm-openai
```

### SGLang
```bash
docker run --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=64g --name sglang1 --ulimit memlock=-1 -dit --network=host --cap-add=IPC_LOCK --privileged --device=/dev/infiniband -v /nfs/gen_media/api.liandanxia:/nfs/gen_media/api.liandanxia -v /data:/data lmsysorg/sglang
```

> **提示**：启动容器后，使用 `docker exec -it <容器名称> bash` 进入容器内，并切换至脚本目录（例如 `cd /data/ai-infer`），再执行下方的 `bench.py` 压测命令。

## 2. 目录结构

```
ai-infer/
├── bench.py              # 统一入口
├── configs/              # YAML 配置文件 (按引擎分目录)
│   ├── vllm/
│   │   ├── deepseek-v3.2.yaml
│   │   ├── deepseek-v3.1-T.yaml
│   │   ├── minimaxm2.5.yaml
│   │   └── qwen3-32b.yaml
│   ├── sglang/
│   │   └── qwen3-32b.yaml
│   └── trtllm/
│       └── qwen3-32b.yaml
├── backends/             # 后端插件 (每个框架一个)
│   ├── base.py           # 抽象基类
│   ├── sglang.py
│   ├── vllm.py
│   └── trtllm.py
├── core/                 # 公共逻辑
│   ├── session.py        # Session 管理
│   ├── server.py         # 服务启动/健康检查/终止
│   ├── benchmark.py      # 压测执行
│   └── metrics.py        # 结果处理/表格打印/汇总
└── logs/                 # 压测日志 (自动生成, git ignored)
    ├── vllm/{模型名}/bench_{时间戳}/
    ├── sglang/{模型名}/bench_{时间戳}/
    └── trtllm/{模型名}/bench_{时间戳}/
```

## 用法

```bash
# 完整压测 (启动服务 + 压测 + 自动 kill)
python3 bench.py -c configs/sglang/qwen3-32b.yaml

# 只启动服务
python3 bench.py -c configs/vllm/qwen3-32b.yaml --server-only

# 只跑压测 (复用已运行的服务)
python3 bench.py -c configs/vllm/qwen3-32b.yaml --no-server

# 查看结果
python3 bench.py -c configs/trtllm/qwen3-32b.yaml --show-summary

# 查看指定 session 的结果
python3 bench.py -c configs/trtllm/qwen3-32b.yaml --show-summary bench_20260302_092317
```

## 功能测试

独立于性能压测的功能正确性验证，通过 OpenAI 兼容 API 调用，sglang / vllm / trtllm 通用。

| 测试类型 | 说明 | 默认样本数 |
|---------|------|----------|
| `tool_calling` | 工具调用 (验证 tool_calls 生成) | 5 |
| `structured_output` | 结构化输出 (验证 JSON Schema 合规) | 5 |
| `gsm8k` | 数学推理 (GSM8K 数据集) | 100 |
| `mmlu` | 多任务理解 (MMLU 数据集) | 50 |

```bash
# 需要服务已运行, 先启动服务
python3 bench.py -c configs/sglang/qwen3-32b.yaml --server-only

# 在另一个终端运行功能测试
python3 bench.py -c configs/sglang/qwen3-32b.yaml --no-server --test tool_calling
python3 bench.py -c configs/sglang/qwen3-32b.yaml --no-server --test structured_output
python3 bench.py -c configs/sglang/qwen3-32b.yaml --no-server --test gsm8k
python3 bench.py -c configs/sglang/qwen3-32b.yaml --no-server --test mmlu --num-samples 20

# 也可以一步启动服务 + 跑测试 (完成后自动 kill 服务)
python3 bench.py -c configs/vllm/qwen3-32b.yaml --test tool_calling
```

依赖: `pip install openai datasets` (gsm8k / mmlu 需要 datasets)

## 新增模型/框架

- **新增模型**: 在 `configs/<引擎>/` 下新建 YAML 配置文件，修改 `model.path`、`model.name` 和 `benchmark.rounds`。
- **新增框架**: 在 `backends/` 下新建文件实现 `BackendBase` 的 3 个方法，然后在 `backends/__init__.py` 的 `BACKEND_REGISTRY` 中注册，并在 `configs/` 下新建对应引擎目录。
