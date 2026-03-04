# AI Inference Benchmark Pipeline

统一压测 Pipeline — 配置驱动 + 插件化后端架构。

## 目录结构

```
ai-infer/
├── bench.py              # 统一入口
├── configs/              # YAML 配置文件 (每个模型×框架一个)
│   ├── sglang_qwen3-32b.yaml
│   ├── vllm_qwen3-32b.yaml
│   └── trtllm_qwen3-32b.yaml
├── backends/             # 后端插件 (每个框架一个)
│   ├── base.py           # 抽象基类
│   ├── sglang.py
│   ├── vllm.py
│   └── trtllm.py
└── core/                 # 公共逻辑
    ├── session.py        # Session 管理
    ├── server.py         # 服务启动/健康检查/终止
    ├── benchmark.py      # 压测执行
    └── metrics.py        # 结果处理/表格打印/汇总
```

## 用法

```bash
# 完整压测 (启动服务 + 压测 + 自动 kill)
python3 bench.py -c configs/sglang_qwen3-32b.yaml

# 只启动服务
python3 bench.py -c configs/vllm_qwen3-32b.yaml --server-only

# 只跑压测 (复用已运行的服务)
python3 bench.py -c configs/vllm_qwen3-32b.yaml --no-server

# 查看结果
python3 bench.py -c configs/trtllm_qwen3-32b.yaml --show-summary

# 查看指定 session 的结果
python3 bench.py -c configs/trtllm_qwen3-32b.yaml --show-summary bench_20260302_092317
```

## 新增模型/框架

- **新增模型**: 复制一个 YAML 配置文件，修改 `model.path`、`model.name` 和 `benchmark.rounds`。
- **新增框架**: 在 `backends/` 下新建文件实现 `BackendBase` 的 3 个方法，然后在 `backends/__init__.py` 的 `BACKEND_REGISTRY` 中注册。
