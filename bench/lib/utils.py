# Ported from vllm/benchmarks/lib/utils.py
# Removed vllm-specific functions (default_vllm_config, PyTorch benchmark format).
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from typing import Any


class InfEncoder(json.JSONEncoder):
    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {
                str(k)
                if not isinstance(k, (str, int, float, bool, type(None)))
                else k: self.clear_inf(v)
                for k, v in o.items()
            }
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o: f"<{type(o).__name__} is not JSON serializable>",
        )
