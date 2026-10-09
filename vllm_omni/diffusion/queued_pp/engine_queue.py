# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-side FIFO records for queued diffusion step Futures."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueuedStepFuture:
    future: Future
    scheduler_output: Any
    request_ids: tuple[str, ...]


class QueuedStepFutureQueue:
    """Bounded FIFO that keeps each request in at most one in-flight step."""

    def __init__(self, depth: int) -> None:
        if type(depth) is not int or depth <= 0:
            raise ValueError(f"queued step Future depth must be positive, got {depth!r}")
        self.depth = depth
        self._entries: deque[QueuedStepFuture] = deque()
        self._request_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def request_ids(self) -> frozenset[str]:
        return frozenset(self._request_ids)

    @property
    def has_capacity(self) -> bool:
        return len(self._entries) < self.depth

    def contains_request(self, request_id: str) -> bool:
        return request_id in self._request_ids

    def append(self, entry: QueuedStepFuture) -> None:
        if not self.has_capacity:
            raise RuntimeError("Queued step Future FIFO is full")
        if not entry.request_ids or self._request_ids.intersection(entry.request_ids):
            raise ValueError("A request may have at most one queued step Future")
        self._entries.appendleft(entry)
        self._request_ids.update(entry.request_ids)

    def pop_oldest(self) -> QueuedStepFuture:
        return self._entries.pop()

    def release(self, entry: QueuedStepFuture) -> None:
        self._request_ids.difference_update(entry.request_ids)
