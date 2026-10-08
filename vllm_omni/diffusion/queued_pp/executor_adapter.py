# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small result adapters shared by queued pipeline Executors.

Collective RPC implementations may add one or more list envelopes around a
rank-aggregated result. The adapters here only remove those envelopes; type
and topology validation remains at each call site.
"""

from __future__ import annotations

from typing import Any


def unwrap_nested_pipeline_result(result: Any) -> Any:
    """Remove singleton list envelopes whose payload is another list."""
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    return result


def unwrap_singleton_pipeline_result(result: Any) -> Any:
    """Remove singleton list envelopes, including scalar payloads."""
    while isinstance(result, list) and len(result) == 1:
        result = result[0]
    return result
