#!/usr/bin/env python3
"""
Universal LLM Inference Benchmark Tool.

A standalone benchmark tool extracted from vLLM, capable of benchmarking
any OpenAI-compatible inference server (vLLM, SGLang, TensorRT-LLM, etc.).

Usage:
    python -m bench serve [options]
    python -m bench serve --help

Examples:
    # Benchmark with random data (simplest, no tokenizer needed)
    python -m bench serve --backend openai-chat \
        --base-url http://localhost:8000 \
        --model Qwen/Qwen2-7B \
        --dataset-name random \
        --num-prompts 100 \
        --skip-tokenizer-init

    # Benchmark with ShareGPT data
    python -m bench serve --backend openai-chat \
        --base-url http://localhost:30000 \
        --model Qwen/Qwen2-7B \
        --dataset-name sharegpt \
        --dataset-path /path/to/ShareGPT_V3.json \
        --num-prompts 500

    # Benchmark with rate limiting
    python -m bench serve --backend openai-chat \
        --base-url http://localhost:8000 \
        --model Qwen/Qwen2-7B \
        --dataset-name random \
        --num-prompts 1000 \
        --request-rate 10 \
        --save-result
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Universal LLM Inference Benchmark Tool",
        usage="python -m bench <command> [options]",
    )
    subparsers = parser.add_subparsers(dest="command", help="Benchmark command")

    # 'serve' subcommand - benchmark online serving
    serve_parser = subparsers.add_parser(
        "serve",
        help="Benchmark online serving throughput and latency.",
        description="Benchmark online serving throughput and latency "
        "against any OpenAI-compatible inference server.",
    )

    from bench.bench_serving import add_cli_args, main as serve_main

    add_cli_args(serve_parser)
    serve_parser.set_defaults(func=serve_main)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
