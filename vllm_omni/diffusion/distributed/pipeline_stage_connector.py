# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, identity-checked transport contract for queued PP edges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PipelineMessage:
    batch_id: str
    step_index: int
    epoch: int
    branch: str
    payload: Any


@dataclass
class TransferTicket:
    message: PipelineMessage
    completed: bool = False
    released: bool = False


class PipelineTransport(Protocol):
    def send(self, message: PipelineMessage) -> Any: ...

    def poll(self) -> list[PipelineMessage]: ...


class PipelineStageConnector:
    """Own one directed edge's bounded send/receive leases.

    The connector does not choose requests, call model code, or create a
    background thread.  A Worker progress loop calls ``poll_received`` and
    explicitly retires tickets after consumer completion.
    """

    def __init__(self, *, edge: str, max_slots: int = 1, transport: PipelineTransport | None = None) -> None:
        if not edge:
            raise ValueError("edge must be non-empty")
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError("max_slots must be a positive integer")
        self.edge = edge
        self.max_slots = max_slots
        self.transport = transport
        self._send_tickets: deque[TransferTicket] = deque()
        self._received: deque[PipelineMessage] = deque()

    @property
    def send_in_use(self) -> int:
        return len(self._send_tickets)

    @property
    def receive_depth(self) -> int:
        return len(self._received)

    def enqueue_send(self, message: PipelineMessage) -> TransferTicket:
        self._validate_message(message)
        if self.send_in_use >= self.max_slots:
            raise RuntimeError(f"pipeline edge {self.edge!r} has no send credit")
        ticket = TransferTicket(message=message)
        self._send_tickets.append(ticket)
        if self.transport is not None:
            self.transport.send(message)
        return ticket

    def mark_send_complete(self, ticket: TransferTicket) -> None:
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        ticket.completed = True

    def poll_received(self) -> list[PipelineMessage]:
        if self.transport is not None:
            for message in self.transport.poll():
                self._validate_message(message)
                self._received.append(message)
        messages = list(self._received)
        self._received.clear()
        return messages

    def release_send(self, ticket: TransferTicket) -> None:
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.completed:
            raise RuntimeError("cannot release a transfer before transport completion")
        self._send_tickets.remove(ticket)
        ticket.released = True

    def _validate_message(self, message: PipelineMessage) -> None:
        if message.branch != "conditional":
            raise ValueError("M2 supports only the conditional pipeline branch")
        if not message.batch_id or message.step_index < 0 or message.epoch < 0:
            raise ValueError("invalid pipeline message identity")
