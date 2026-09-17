# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, identity-checked transport contract for queued PP edges."""

from __future__ import annotations

import inspect
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

    def poll(self, limit: int | None = None) -> list[PipelineMessage]: ...

    def wait(self, ticket: TransferTicket) -> bool: ...

    def abort(self, ticket: TransferTicket) -> bool: ...

    def close(self) -> None: ...


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
        self._transport_pending: deque[PipelineMessage] = deque()
        self._received_leases: dict[int, PipelineMessage] = {}
        self._closed = False

    @property
    def send_in_use(self) -> int:
        return len(self._send_tickets)

    @property
    def receive_depth(self) -> int:
        return len(self._received) + len(self._received_leases)

    @property
    def transport_pending(self) -> int:
        return len(self._transport_pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def enqueue_send(self, message: PipelineMessage) -> TransferTicket:
        self._ensure_open()
        self._validate_message(message)
        if self.send_in_use >= self.max_slots:
            raise RuntimeError(f"pipeline edge {self.edge!r} has no send credit")
        ticket = TransferTicket(message=message)
        self._send_tickets.append(ticket)
        try:
            if self.transport is not None:
                self.transport.send(message)
        except BaseException:
            self._send_tickets.remove(ticket)
            raise
        return ticket

    def mark_send_complete(self, ticket: TransferTicket) -> None:
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        ticket.completed = True

    def poll_received(self, limit: int = 1) -> list[PipelineMessage]:
        self._ensure_open()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        available = self.max_slots - self.receive_depth - self.transport_pending
        if self.transport is not None and available > 0 and not self._transport_pending:
            poll_limit = min(limit, available)
            parameters = inspect.signature(self.transport.poll).parameters.values()
            supports_limit = any(
                parameter.name == "limit" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
            )
            if not supports_limit:
                raise RuntimeError("pipeline transport must support bounded polling") from None
            incoming = self.transport.poll(limit=poll_limit)
            if len(incoming) > poll_limit:
                raise RuntimeError("pipeline transport exceeded its bounded poll limit")
            incoming_keys = [self._message_identity(message) for message in incoming]
            held_keys = {
                self._message_identity(message)
                for message in (*self._received, *self._transport_pending, *self._received_leases.values())
            }
            if len(set(incoming_keys)) != len(incoming_keys) or held_keys.intersection(incoming_keys):
                raise RuntimeError("pipeline transport returned a duplicate leased message")
            for message in incoming:
                self._validate_message(message)
            self._transport_pending.extend(incoming)
        while self._transport_pending and len(self._received) < self.max_slots:
            self._received.append(self._transport_pending.popleft())
        messages = [self._received.popleft() for _ in range(min(limit, len(self._received)))]
        for message in messages:
            self._received_leases[id(message)] = message
        return messages

    def release_received(self, message: PipelineMessage) -> None:
        """Return receive credit after the consumer is finished reading."""
        self._ensure_open()
        if self._received_leases.pop(id(message), None) is None:
            raise ValueError("unknown or already released receive lease")

    def release_send(self, ticket: TransferTicket) -> None:
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.completed:
            raise RuntimeError("cannot release a transfer before transport completion")
        self._send_tickets.remove(ticket)
        ticket.released = True

    def retire_batch(self, batch_id: str, *, discard_results: bool = False) -> None:
        """Retire all local transport state for one batch after dependencies settle."""
        self._ensure_open()
        if not batch_id:
            raise ValueError("batch_id must be non-empty")
        matching = [ticket for ticket in self._send_tickets if ticket.message.batch_id == batch_id]
        leased = [message for message in self._received_leases.values() if message.batch_id == batch_id]
        if leased:
            raise RuntimeError("cannot retire a batch before receive consumers complete")
        if matching and not discard_results:
            unreleased = [ticket for ticket in matching if not ticket.completed]
            if unreleased:
                raise RuntimeError("cannot retire a batch before transport completion")
        for ticket in matching:
            if not ticket.completed:
                if not discard_results:
                    raise RuntimeError("cannot retire a batch before transport completion")
                self._wait_or_abort(ticket, discard=True)
        for ticket in matching:
            self._send_tickets.remove(ticket)
            ticket.released = True
        self._received = deque(message for message in self._received if message.batch_id != batch_id)
        self._transport_pending = deque(message for message in self._transport_pending if message.batch_id != batch_id)

    def health(self) -> dict[str, Any]:
        return {
            "edge": self.edge,
            "closed": self._closed,
            "send_in_use": self.send_in_use,
            "receive_depth": self.receive_depth,
            "receive_leases": len(self._received_leases),
            "transport_pending": self.transport_pending,
            "max_slots": self.max_slots,
        }

    def close(self, *, drain: bool = False) -> None:
        if self._closed:
            return
        if self._received_leases:
            raise RuntimeError("cannot close a connector with active receive consumers")
        if self._send_tickets and not drain:
            raise RuntimeError("cannot close a connector with outstanding transfers")
        if drain:
            for ticket in self._send_tickets:
                if not ticket.completed:
                    self._wait_or_abort(ticket, discard=False)
        transport_close = getattr(self.transport, "close", None)
        if self.transport is not None and not callable(transport_close):
            raise RuntimeError("pipeline transport does not support close")
        if transport_close is not None:
            transport_close()
        for ticket in self._send_tickets:
            ticket.released = True
        self._send_tickets.clear()
        self._received.clear()
        self._received_leases.clear()
        self._transport_pending.clear()
        self._closed = True

    def _wait_or_abort(self, ticket: TransferTicket, *, discard: bool) -> None:
        operation = "abort" if discard else "wait"
        handler = getattr(self.transport, operation, None)
        if not callable(handler):
            raise RuntimeError(f"transport does not support {operation} for incomplete transfers")
        if not handler(ticket):
            raise RuntimeError(f"transport {operation} did not complete transfer")
        ticket.completed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"pipeline edge {self.edge!r} is closed")

    def _validate_message(self, message: PipelineMessage) -> None:
        if message.branch != "conditional":
            raise ValueError("M2 supports only the conditional pipeline branch")
        if not message.batch_id or message.step_index < 0 or message.epoch < 0:
            raise ValueError("invalid pipeline message identity")

    @staticmethod
    def _message_identity(message: PipelineMessage) -> tuple[str, int, int, str]:
        return (message.batch_id, message.step_index, message.epoch, message.branch)
