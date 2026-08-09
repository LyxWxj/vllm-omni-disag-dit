# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit request-row mappings for step-level diffusion execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RequestRowLayout:
    """Map physical tensor rows back to request-local prediction rows.

    ``request_ids`` defines the logical request namespace. For every physical
    row, ``row_to_request`` stores an index into that namespace and
    ``row_to_request_row`` stores the row's position in the request-local
    prediction tensor. Physical rows may be reordered or interleaved.
    """

    request_ids: tuple[str, ...]
    row_to_request: tuple[int, ...]
    row_to_request_row: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("RequestRowLayout requires at least one request.")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("RequestRowLayout request_ids must be unique.")
        if len(self.row_to_request) != len(self.row_to_request_row):
            raise ValueError("row_to_request and row_to_request_row must have the same length.")
        if not self.row_to_request:
            raise ValueError("RequestRowLayout requires at least one row.")

        local_rows: list[list[int]] = [[] for _ in self.request_ids]
        for physical_row, (request_index, request_row) in enumerate(
            zip(self.row_to_request, self.row_to_request_row, strict=True)
        ):
            if request_index < 0 or request_index >= len(self.request_ids):
                raise ValueError(
                    f"row_to_request[{physical_row}]={request_index} is out of range for "
                    f"{len(self.request_ids)} requests."
                )
            if request_row < 0:
                raise ValueError(f"row_to_request_row[{physical_row}] must be non-negative.")
            local_rows[request_index].append(request_row)

        for request_index, rows in enumerate(local_rows):
            if not rows:
                raise ValueError(f"Request {self.request_ids[request_index]!r} must own at least one row.")
            expected = list(range(len(rows)))
            if sorted(rows) != expected:
                raise ValueError(
                    f"Rows for request {self.request_ids[request_index]!r} must be a permutation of "
                    f"{expected}; got {rows}."
                )

    @classmethod
    def from_request_row_counts(
        cls,
        request_ids: Sequence[str],
        row_counts: Sequence[int],
    ) -> RequestRowLayout:
        """Build the contiguous identity layout for request-local row counts."""
        if len(request_ids) != len(row_counts):
            raise ValueError("request_ids and row_counts must have the same length.")

        row_to_request: list[int] = []
        row_to_request_row: list[int] = []
        for request_index, row_count in enumerate(row_counts):
            if row_count <= 0:
                raise ValueError(f"row_counts[{request_index}] must be positive; got {row_count}.")
            row_to_request.extend([request_index] * row_count)
            row_to_request_row.extend(range(row_count))
        return cls(
            request_ids=tuple(request_ids),
            row_to_request=tuple(row_to_request),
            row_to_request_row=tuple(row_to_request_row),
        )

    @property
    def num_rows(self) -> int:
        return len(self.row_to_request)

    def request_row_count(self, request_id: str) -> int:
        request_index = self._request_index(request_id)
        return sum(owner == request_index for owner in self.row_to_request)

    def split_tensor(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split and locally reorder a physical-row tensor by request ID."""
        if value.ndim == 0:
            raise ValueError("Cannot split a scalar tensor with RequestRowLayout.")
        if int(value.shape[0]) != self.num_rows:
            raise ValueError(f"RequestRowLayout describes {self.num_rows} rows, but tensor has {int(value.shape[0])}.")

        physical_rows: list[list[tuple[int, int]]] = [[] for _ in self.request_ids]
        for physical_row, (request_index, request_row) in enumerate(
            zip(self.row_to_request, self.row_to_request_row, strict=True)
        ):
            physical_rows[request_index].append((request_row, physical_row))

        result: dict[str, torch.Tensor] = {}
        for request_id, rows in zip(self.request_ids, physical_rows, strict=True):
            ordered_indices = [physical_row for _, physical_row in sorted(rows)]
            index = torch.tensor(ordered_indices, dtype=torch.long, device=value.device)
            result[request_id] = value.index_select(0, index)
        return result

    def validate_compatible(self, other: RequestRowLayout) -> None:
        """Require both layouts to describe the same request-local rows."""
        if set(self.request_ids) != set(other.request_ids):
            raise ValueError(f"RequestRowLayout request IDs do not match: {self.request_ids} != {other.request_ids}.")
        for request_id in self.request_ids:
            row_count = self.request_row_count(request_id)
            other_row_count = other.request_row_count(request_id)
            if row_count != other_row_count:
                raise ValueError(
                    f"RequestRowLayout row count for {request_id!r} does not match: {row_count} != {other_row_count}."
                )

    def _request_index(self, request_id: str) -> int:
        try:
            return self.request_ids.index(request_id)
        except ValueError as exc:
            raise KeyError(f"Unknown request ID in RequestRowLayout: {request_id!r}.") from exc


@dataclass(frozen=True)
class DenoiseStepOutput:
    """A denoise prediction with an explicit physical-to-request row map."""

    prediction: torch.Tensor | None
    row_layout: RequestRowLayout
