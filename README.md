# ai-infer

本项目主要用于记录大模型推理（Large Language Model Inference）以及相关压测（Benchmark）的脚本和资料。

## 目录结构

### `scripts/`
包含多个主流推理框架（vLLM, SGLang, TensorRT-LLM）的自动化多轮测试 Pipeline。支持：
- 服务生命周期自动管理（后台启动、健康检查、自动终止）
- 多轮自动逐级压测，并支持 `session` 机制重用/合并历史测试结果
- 多种压测场景（Prefill 长输入短输出、Decode 短输入长输出、Mixed 混合流量等）
- 自动提取关键性能指标并汇总记录

当前包含适用于 `Qwen3-32B` 的压测脚本：
- `vllm_pipeline_qwen3-32b.py`
- `sglang_pipeline_qwen3-32b.py`
- `trtllm_pipeline_qwen3-32b.py`

## 使用说明

各个框架压测脚本通常包含如下参数和运行模式：
```bash
# 完整压测流程（启动服务 -> 多轮测试 -> 提取结果 -> 停止服务）
python scripts/<framework>_pipeline_qwen3-32b.py

# 组合运行：独立启动服务端，然后运行独立的客户端
# 比如服务端可以单独监控，客户端可修改并发参数反复压测
## 1. 仅启动服务端：
python scripts/<framework>_pipeline_qwen3-32b.py --server-only
## 2. 仅运行压测（不启动服务端）：
python scripts/<framework>_pipeline_qwen3-32b.py --no-server
## 3. 最后手动终止服务端。

# 打印过往的 session 汇总数据
python scripts/<framework>_pipeline_qwen3-32b.py --show-summary
```
