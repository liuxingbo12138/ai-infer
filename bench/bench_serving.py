from __future__ import annotations

# Core benchmark serving logic.
# Ported from vllm/benchmarks/serve.py - all vllm dependencies removed.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

r"""Benchmark online serving throughput.

Run against any OpenAI-compatible inference server (vLLM, SGLang, TensorRT-LLM, etc.):

    python -m bench serve \
        --backend openai-chat \
        --base-url http://localhost:8000 \
        --model <your_model> \
        --dataset-name random \
        --num-prompts 1000
"""

import argparse
import asyncio
import contextlib
import gc
import importlib.util
import json
import os
import random
import shutil
import ssl
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from bench.datasets import SampleRequest, add_dataset_args, get_samples
from bench.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from bench.lib.ready_checker import wait_for_endpoint

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = (importlib.util.find_spec("termplotlib") is not None) and (
    shutil.which("gnuplot") is not None
)


def join_host_port(host: str, port: int) -> str:
    """Join host and port, handling IPv6 addresses."""
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


async def get_first_model_from_server(
    base_url: str,
    headers: dict | None = None,
    ssl_context: ssl.SSLContext | bool | None = None,
) -> tuple[str, str]:
    """Fetch the first model from the server's /v1/models endpoint."""
    models_url = f"{base_url}/v1/models"
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.get(models_url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"], data["data"][0]["root"]
                else:
                    raise ValueError(
                        f"No models found on the server at {base_url}. "
                        "Make sure the server is running and has models loaded."
                    )
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to fetch models from server at {models_url}. "
                "Check that:\n"
                "1. The server is running\n"
                "2. The server URL is correct\n"
                f"Error: {e}"
            ) from e


class TaskType(Enum):
    GENERATION = "generation"
    POOLING = "pooling"


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    max_output_tokens_per_s: float
    max_concurrent_requests: int


@dataclass
class EmbedBenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    mean_e2el_ms: float
    std_e2el_ms: float
    median_e2el_ms: float
    percentiles_e2el_ms: float


def _get_current_request_rate(
    ramp_up_strategy: Literal["linear", "exponential"] | None,
    ramp_up_start_rps: int | None,
    ramp_up_end_rps: int | None,
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if (
        ramp_up_strategy
        and ramp_up_start_rps is not None
        and ramp_up_end_rps is not None
    ):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        elif ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        else:
            raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness and OPTIONAL ramp-up strategy.
    """
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    if isinstance(input_requests, Iterable) and not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    # Precompute delays among requests
    request_rates = []
    delay_ts = []
    for request_index, request in enumerate(input_requests):
        current_request_rate = _get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )
        assert current_request_rate > 0.0, (
            f"Obtained non-positive request rate {current_request_rate}."
        )
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0)
        elif burstiness == float("inf"):
            delay_ts.append(1.0 / current_request_rate)
        else:
            theta = 1.0 / (current_request_rate * burstiness)
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    # Calculate cumulative delay time
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if ramp_up_strategy is None and delay_ts[-1] != 0:
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for request_index, request in enumerate(input_requests):
        if delay_ts[request_index] > 0:
            current_ts = time.time()
            sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]


def calculate_metrics_for_embeddings(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
) -> EmbedBenchmarkMetrics:
    """Calculate the metrics for embedding requests."""
    total_input = 0
    completed = 0
    failed = 0
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            e2els.append(outputs[i].latency)
            completed += 1
            total_input += outputs[i].prompt_len
        else:
            failed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = EmbedBenchmarkMetrics(
        completed=completed,
        failed=failed,
        total_input=total_input,
        request_throughput=completed / dur_s,
        total_token_throughput=total_input / dur_s,
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )
    return metrics


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: Any,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark."""
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                if tokenizer is None:
                    output_len = 1
                else:
                    output_len = len(
                        tokenizer(
                            outputs[i].generated_text, add_special_tokens=False
                        ).input_ids
                    )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate max output tokens per second metric
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]

    if len(failed_outputs) > 0:
        print("Failed requests during benchmark run detected (capping to 10):")
        for i, err in enumerate(failed_outputs[:10]):
            print(f"Error {i}: {err.error}")

    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(
            output.start_time + output.latency for output in successful_outputs
        )

        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for i, output in enumerate(successful_outputs):
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int(
                (output.start_time + output.latency) - min_start_time
            )

            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1

        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if TERM_PLOTLIB_AVAILABLE:
            import termplotlib as tpl

            fig = tpl.figure()
            fig.plot(
                np.arange(len(tokens_per_second)),
                tokens_per_second,
                title="Output tokens per second",
            )
            fig.plot(
                np.arange(len(concurrent_requests_per_second)),
                concurrent_requests_per_second,
                title="Concurrent requests per second",
            )
            fig.show()
        else:
            print("tip: install termplotlib and gnuplot to plot the metrics")

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )

    return metrics, actual_output_lens


async def benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: Any,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    extra_headers: dict | None,
    extra_body: dict | None,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
    ssl_context: ssl.SSLContext | bool | None = None,
):
    try:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    except KeyError:
        raise ValueError(f"Unknown backend: {endpoint_type}") from None

    # Reuses connections across requests to reduce TLS handshake overhead.
    ssl_setting = ssl_context if ssl_context is not None else ("https://" in api_url)
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=ssl_setting,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )

    if ready_check_timeout_sec > 0:
        test_output = await wait_for_endpoint(
            request_func,
            test_input,
            session,
            timeout_seconds=ready_check_timeout_sec,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark "
                "arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        else:
            print("Initial test run completed.")
    else:
        print("Skipping endpoint ready check.")

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )
        warmup_tasks = []

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(
                    request_func_input=test_input, session=session, pbar=warmup_pbar
                )

        for _ in range(num_warmups):
            request_task = asyncio.create_task(warmup_limited_request_func())
            warmup_tasks.append(request_task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup run completed.")

    print("Starting main benchmark run...")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    if ramp_up_strategy is not None:
        print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
        print(
            f"Will increase RPS from {ramp_up_start_rps} to "
            f"{ramp_up_end_rps} RPS over the duration of the benchmark."
        )
    else:
        print(f"Traffic request rate: {request_rate}")

    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request_func(request_func_input, session, pbar):
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input, session=session, pbar=pbar
            )

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []

    async for request, current_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        request_func_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            logprobs=logprobs,
            multi_modal_content=request.multi_modal_data,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
            request_id=request.request_id,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input, session, pbar)
            )
        )

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    await session.close()

    # Build the result dict
    result: dict[str, Any] = {}

    if task_type == TaskType.POOLING:
        metrics = calculate_metrics_for_embeddings(
            outputs=outputs,
            dur_s=benchmark_duration,
            selected_percentiles=[
                float(p) for p in selected_percentiles
            ],
        )
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "failed": metrics.failed,
            "total_input": metrics.total_input,
            "request_throughput": metrics.request_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "mean_e2el_ms": metrics.mean_e2el_ms,
            "median_e2el_ms": metrics.median_e2el_ms,
            "std_e2el_ms": metrics.std_e2el_ms,
        }
    else:
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            selected_percentiles=[
                float(p) for p in selected_percentiles
            ],
            goodput_config_dict=goodput_config_dict,
        )

        print("\n" + "=" * 60)
        print("  Benchmark Results")
        print("=" * 60)
        print(f"  Successful requests:     {metrics.completed:>10}")
        print(f"  Failed requests:         {metrics.failed:>10}")
        print(f"  Benchmark duration (s):  {benchmark_duration:>10.2f}")
        print(f"  Total input tokens:      {metrics.total_input:>10}")
        print(f"  Total generated tokens:  {metrics.total_output:>10}")
        print(f"  Request throughput (req/s):{metrics.request_throughput:>9.2f}")
        print(f"  Output token throughput (tok/s):{metrics.output_throughput:>5.2f}")
        print(f"  Total token throughput (tok/s):{metrics.total_token_throughput:>6.2f}")

        def _print_metric(header, mean, median, std, percentiles):
            print(f"\n  {header}:")
            print(f"    Mean:   {mean:>10.2f} ms")
            print(f"    Median: {median:>10.2f} ms")
            print(f"    Std:    {std:>10.2f} ms")
            for p, val in percentiles:
                print(f"    P{p:<4}:  {val:>10.2f} ms")

        if "ttft" in selected_percentile_metrics:
            _print_metric(
                "Time to First Token (TTFT)",
                metrics.mean_ttft_ms,
                metrics.median_ttft_ms,
                metrics.std_ttft_ms,
                metrics.percentiles_ttft_ms,
            )
        if "tpot" in selected_percentile_metrics:
            _print_metric(
                "Time per Output Token (TPOT)",
                metrics.mean_tpot_ms,
                metrics.median_tpot_ms,
                metrics.std_tpot_ms,
                metrics.percentiles_tpot_ms,
            )
        if "itl" in selected_percentile_metrics:
            _print_metric(
                "Inter-Token Latency (ITL)",
                metrics.mean_itl_ms,
                metrics.median_itl_ms,
                metrics.std_itl_ms,
                metrics.percentiles_itl_ms,
            )
        if "e2el" in selected_percentile_metrics:
            _print_metric(
                "End-to-End Latency (E2EL)",
                metrics.mean_e2el_ms,
                metrics.median_e2el_ms,
                metrics.std_e2el_ms,
                metrics.percentiles_e2el_ms,
            )

        print(f"\n  Max output tokens/s:     {metrics.max_output_tokens_per_s:>10.2f}")
        print(f"  Max concurrent requests: {metrics.max_concurrent_requests:>10}")
        print("=" * 60)

        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "failed": metrics.failed,
            "total_input": metrics.total_input,
            "total_output": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "mean_e2el_ms": metrics.mean_e2el_ms,
            "median_e2el_ms": metrics.median_e2el_ms,
            "std_e2el_ms": metrics.std_e2el_ms,
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
            "input_lens": [req.prompt_len for req in input_requests],
            "output_lens": actual_output_lens,
            "ttfts": [o.ttft for o in outputs],
            "itls": [o.itl for o in outputs],
            "generated_texts": [o.generated_text for o in outputs],
            "errors": [o.error for o in outputs],
        }

        # Add percentiles
        for p, val in metrics.percentiles_ttft_ms:
            result[f"p{int(p)}_ttft_ms"] = val
        for p, val in metrics.percentiles_tpot_ms:
            result[f"p{int(p)}_tpot_ms"] = val
        for p, val in metrics.percentiles_itl_ms:
            result[f"p{int(p)}_itl_ms"] = val
        for p, val in metrics.percentiles_e2el_ms:
            result[f"p{int(p)}_e2el_ms"] = val

    return result


# ============================================================================
# CLI argument parser
# ============================================================================


def check_goodput_args(args):
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def add_cli_args(parser: argparse.ArgumentParser):
    add_dataset_args(parser)
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label (prefix) of the benchmark results.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai-chat",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="The type of backend or endpoint to use for the benchmark.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--header",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs for headers to be passed with each request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Name of the model. If not specified, will fetch the first model "
        "from the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="General input length for datasets.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="General output length for datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Number of logprobs-per-token to compute & return.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If inf, all requests sent at time 0.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Default 1 follows Poisson process.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to a json file.",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving results, include per-request information.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs for metadata of this run.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save benchmark json results.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Filename to save benchmark json results.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default=None,
        help="Comma-separated list of selected metrics to report percentiles. "
        'Allowed: "ttft", "tpot", "itl", "e2el".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help='Comma-separated list of percentiles. Default "99".',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify SLOs for goodput as "KEY:VALUE" pairs.',
    )
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        required=False,
        default=f"bench-{uuid.uuid4().hex[:8]}-",
        help="Prefix of request id.",
    )
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)
    sampling_group.add_argument("--frequency-penalty", type=float, default=None)
    sampling_group.add_argument("--presence-penalty", type=float, default=None)
    sampling_group.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API.",
    )
    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy for request rate.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="Starting request rate for ramp-up (RPS).",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="Ending request rate for ramp-up (RPS).",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=0,
        help="Maximum time to wait for endpoint readiness in seconds. "
        "Ready check will be skipped by default.",
    )
    parser.add_argument(
        "--extra-body",
        help="A JSON string representing extra body parameters.",
        type=json.loads,
        default=None,
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        default=False,
        help="Skip initialization of tokenizer and detokenizer.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=False,
        help="Disable SSL certificate verification.",
    )


# ============================================================================
# Main entry
# ============================================================================


def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

    label = args.label

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    # Headers
    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")

    # SSL context configuration
    ssl_context: ssl.SSLContext | bool | None = None
    if args.insecure:
        ssl_context = False
    elif "https://" in base_url:
        ssl_context = True

    # Fetch model from server if not specified
    if args.model is None:
        print("Model not specified, fetching first model from server...")
        model_name, model_id = await get_first_model_from_server(
            base_url, headers, ssl_context
        )
        print(f"First model name: {model_name}, first model id: {model_id}")
    else:
        model_name = args.served_model_name
        model_id = args.model

    if args.skip_tokenizer_init:
        tokenizer = None
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer_id = args.tokenizer if args.tokenizer is not None else model_id
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=args.trust_remote_code,
            )
        except ImportError:
            print(
                "WARNING: transformers not installed, running without tokenizer. "
                "Token counts may be inaccurate."
            )
            tokenizer = None
        except Exception as e:
            print(
                f"WARNING: Failed to load tokenizer: {e}. "
                "Running without tokenizer."
            )
            tokenizer = None

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # Map general --input-len and --output-len to dataset-specific arguments
    if args.input_len is not None:
        args.random_input_len = args.input_len

    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sharegpt_output_len = args.output_len
        args.hf_output_len = args.output_len

    # When using random datasets with openai-compatible backends, default to ignoring EOS
    if args.dataset_name == "random" and args.backend in OPENAI_COMPATIBLE_BACKENDS:
        args.ignore_eos = True

    # Load the dataset
    input_requests = get_samples(args, tokenizer)
    goodput_config_dict = check_goodput_args(args)

    backend = args.backend
    task_type = (
        TaskType.POOLING
        if "embeddings" in backend or "rerank" in backend
        else TaskType.GENERATION
    )

    # Collect sampling parameters
    if task_type == TaskType.GENERATION:
        sampling_params = {
            k: v
            for k, v in {
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "temperature": args.temperature,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
            }.items()
            if v is not None
        }

        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError(
                "Sampling parameters are only supported by openai-compatible backends."
            )

        default_percentile_metrics = "ttft,tpot,itl"
    else:
        sampling_params = {}
        default_percentile_metrics = "e2el"

    extra_body = args.extra_body or {}
    extra_body = {**sampling_params, **extra_body}

    percentile_metrics: str = args.percentile_metrics or default_percentile_metrics

    # Disable GC during benchmark for more stable latency measurements
    gc.disable()

    try:
        benchmark_result = await benchmark(
            task_type=task_type,
            endpoint_type=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            num_warmups=args.num_warmups,
            selected_percentile_metrics=percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            extra_headers=headers,
            extra_body=extra_body,
            ramp_up_strategy=args.ramp_up_strategy,
            ramp_up_start_rps=args.ramp_up_start_rps,
            ramp_up_end_rps=args.ramp_up_end_rps,
            ready_check_timeout_sec=args.ready_check_timeout_sec,
            ssl_context=ssl_context,
        )
    finally:
        gc.enable()

    # Save config and results to json
    result_json: dict[str, Any] = {}

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["backend"] = args.backend
    result_json["label"] = label
    result_json["model_id"] = model_id
    result_json["num_prompts"] = args.num_prompts

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=", 1)
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = args.burstiness
    result_json["max_concurrency"] = args.max_concurrency

    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    if not args.save_detailed:
        for field in [
            "input_lens", "output_lens", "start_times",
            "ttfts", "itls", "generated_texts", "errors",
        ]:
            result_json.pop(field, None)

    # Save to file
    if args.save_result or args.append_result:
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        label = label or args.backend
        if args.ramp_up_strategy is not None:
            file_name = (
                f"{label}-ramp-up-{args.ramp_up_strategy}-"
                f"{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps"
                f"{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            )
        else:
            file_name = (
                f"{label}-{args.request_rate}qps"
                f"{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            )
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)

    return result_json
