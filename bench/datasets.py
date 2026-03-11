from __future__ import annotations

# Simplified dataset sampling for benchmarks.
# Ported from vllm/benchmarks/datasets.py - removed LoRA, multimodal, and
# vllm-specific dependencies. Kept: random, sharegpt, hf datasets.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Dataset sampling for LLM inference benchmarks.

Supports three dataset types:
  - random:    Synthetic random token sequences
  - sharegpt:  ShareGPT conversation dataset
  - hf:        Any HuggingFace text dataset
"""

import json
import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SampleRequest:
    """Represents a single inference request for benchmarking."""

    prompt: str | list[str]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: dict | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Random dataset
# ---------------------------------------------------------------------------


def sample_random_requests(
    num_requests: int,
    tokenizer: Any,
    input_len: int = 1024,
    output_len: int = 128,
    prefix_len: int = 0,
    request_id_prefix: str = "",
) -> list[SampleRequest]:
    """Generate random token sequence requests."""
    requests = []

    if tokenizer is not None:
        vocab_size = tokenizer.vocab_size
        for i in range(num_requests):
            token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=f"{request_id_prefix}{i}" if request_id_prefix else None,
                )
            )
    else:
        # Without tokenizer, generate placeholder text
        for i in range(num_requests):
            prompt = "hello " * (input_len // 2)
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=f"{request_id_prefix}{i}" if request_id_prefix else None,
                )
            )

    return requests


# ---------------------------------------------------------------------------
# ShareGPT dataset
# ---------------------------------------------------------------------------


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Any,
    output_len: int | None = None,
    request_id_prefix: str = "",
) -> list[SampleRequest]:
    """Sample requests from a ShareGPT dataset."""
    # Load the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out conversations with less than 2 turns
    dataset = [
        data for data in dataset
        if len(data.get("conversations", [])) >= 2
    ]

    # Shuffle and sample
    random.shuffle(dataset)

    requests = []
    for i, data in enumerate(dataset):
        if len(requests) >= num_requests:
            break

        conversations = data["conversations"]
        # Use the first human turn as input
        input_text = conversations[0].get("value", "")
        if not input_text:
            continue

        if tokenizer is not None:
            input_ids = tokenizer(input_text).input_ids
            prompt_len = len(input_ids)
        else:
            # Rough estimate
            prompt_len = len(input_text.split())

        # Use the first assistant turn for output length estimation
        if output_len is not None:
            expected_output_len = output_len
        elif len(conversations) > 1:
            assistant_text = conversations[1].get("value", "")
            if tokenizer is not None:
                expected_output_len = len(tokenizer(assistant_text).input_ids)
            else:
                expected_output_len = len(assistant_text.split())
        else:
            expected_output_len = 128

        # Basic filtering
        if prompt_len < 4 or expected_output_len < 1:
            continue
        if prompt_len > 4096 or (prompt_len + expected_output_len) > 8192:
            continue

        requests.append(
            SampleRequest(
                prompt=input_text,
                prompt_len=prompt_len,
                expected_output_len=expected_output_len,
                request_id=f"{request_id_prefix}{i}" if request_id_prefix else None,
            )
        )

    if len(requests) < num_requests:
        logger.warning(
            "Only sampled %d requests from ShareGPT dataset (requested %d). "
            "Will oversample to reach target.",
            len(requests),
            num_requests,
        )
        # Oversample
        while len(requests) < num_requests:
            idx = random.randint(0, len(requests) - 1)
            req = requests[idx]
            requests.append(
                SampleRequest(
                    prompt=req.prompt,
                    prompt_len=req.prompt_len,
                    expected_output_len=req.expected_output_len,
                    request_id=f"{request_id_prefix}{len(requests)}"
                    if request_id_prefix
                    else None,
                )
            )

    return requests[:num_requests]


# ---------------------------------------------------------------------------
# HuggingFace dataset
# ---------------------------------------------------------------------------


def sample_hf_requests(
    dataset_name: str,
    dataset_path: str | None,
    num_requests: int,
    tokenizer: Any,
    output_len: int = 128,
    request_id_prefix: str = "",
) -> list[SampleRequest]:
    """Sample requests from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace datasets. "
            "Install it with: pip install datasets"
        ) from exc

    if dataset_path:
        ds = load_dataset(dataset_path, split="train")
    else:
        ds = load_dataset(dataset_name, split="train")

    ds = ds.shuffle(seed=42)

    # Try to find a text column
    text_columns = ["text", "content", "prompt", "question", "input"]
    text_col = None
    for col in text_columns:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = ds.column_names[0]
        logger.warning(
            "No standard text column found, using '%s'", text_col
        )

    requests = []
    for i, row in enumerate(ds):
        if len(requests) >= num_requests:
            break

        text = str(row[text_col])
        if not text or len(text) < 10:
            continue

        if tokenizer is not None:
            input_ids = tokenizer(text, truncation=True, max_length=4096).input_ids
            prompt_len = len(input_ids)
        else:
            prompt_len = len(text.split())

        if prompt_len < 4:
            continue

        requests.append(
            SampleRequest(
                prompt=text,
                prompt_len=prompt_len,
                expected_output_len=output_len,
                request_id=f"{request_id_prefix}{i}" if request_id_prefix else None,
            )
        )

    return requests[:num_requests]


# ---------------------------------------------------------------------------
# Dataset arg parser and dispatcher
# ---------------------------------------------------------------------------


def add_dataset_args(parser) -> None:
    """Add dataset-related CLI arguments."""
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "sharegpt", "hf"],
        help="Name of the dataset to use for benchmarking.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (for sharegpt) or HuggingFace dataset name.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code for tokenizer.",
    )
    # Random dataset args
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Input length for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Output length for random dataset.",
    )
    parser.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Prefix length for random dataset.",
    )
    # ShareGPT dataset args
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override output length for ShareGPT dataset.",
    )
    # HF dataset args
    parser.add_argument(
        "--hf-output-len",
        type=int,
        default=128,
        help="Output length for HuggingFace dataset.",
    )


def get_samples(args, tokenizer) -> list[SampleRequest]:
    """Get sample requests based on dataset arguments."""
    request_id_prefix = getattr(args, "request_id_prefix", "")

    if args.dataset_name == "random":
        return sample_random_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            prefix_len=args.random_prefix_len,
            request_id_prefix=request_id_prefix,
        )
    elif args.dataset_name == "sharegpt":
        if not args.dataset_path:
            raise ValueError(
                "ShareGPT dataset requires --dataset-path pointing to the JSON file."
            )
        return sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.sharegpt_output_len,
            request_id_prefix=request_id_prefix,
        )
    elif args.dataset_name == "hf":
        return sample_hf_requests(
            dataset_name=args.dataset_path or "",
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=request_id_prefix,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
