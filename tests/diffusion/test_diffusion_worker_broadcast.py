# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from unittest.mock import Mock

import pytest
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

import vllm_omni.diffusion.worker.diffusion_worker as diffusion_worker_module
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DelayedMultipartSocket:
    def __init__(
        self,
        socket,
        send_started: threading.Event,
        release_send: threading.Event,
    ):
        self._socket = socket
        self._send_started = send_started
        self._release_send = release_send
        self._delay_next_send = True

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def send_multipart(self, *args, **kwargs):
        if self._delay_next_send:
            self._delay_next_send = False
            self._send_started.set()
            if not self._release_send.wait(timeout=2):
                raise TimeoutError("Timed out waiting to release delayed overflow payload")
        return self._socket.send_multipart(*args, **kwargs)


class _ObservedPollSocket:
    def __init__(self, socket, poll_started: threading.Event):
        self._socket = socket
        self._poll_started = poll_started

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def poll(self, *args, **kwargs):
        self._poll_started.set()
        return self._socket.poll(*args, **kwargs)


class _RecordingWorker:
    def __init__(self):
        self.calls: list[str] = []

    def execute_method(self, method, *args, **kwargs):
        assert method == "record"
        self.calls.append(args[0])


def test_recv_message_waits_indefinitely():
    worker_proc = WorkerProc.__new__(WorkerProc)
    worker_proc.mq = Mock()
    worker_proc.mq.dequeue.return_value = {"type": "rpc"}

    result = worker_proc.recv_message()

    assert result == {"type": "rpc"}
    worker_proc.mq.dequeue.assert_called_once_with(indefinite=True)


def test_busy_loop_releases_failed_rpc_traceback_and_device_cache(monkeypatch):
    original = RuntimeError("model execution failed")
    worker_proc = WorkerProc.__new__(WorkerProc)
    worker_proc.gpu_id = 0
    worker_proc.mq = Mock()
    worker_proc.mq.dequeue.side_effect = [
        {
            "type": "rpc",
            "method": "execute_model",
            "args": (),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": False,
            "collect_rank_status": False,
        },
        {"type": "shutdown"},
    ]
    worker_proc.worker = Mock()
    worker_proc.worker.execute_method.side_effect = original
    worker_proc.result_mq = object()
    worker_proc._return_result = Mock()
    worker_proc.wake_event = None
    worker_proc._running = True

    gc_collect = Mock()
    mock_platform = Mock()
    monkeypatch.setattr(diffusion_worker_module.gc, "collect", gc_collect)
    monkeypatch.setattr(diffusion_worker_module, "current_omni_platform", mock_platform)

    worker_proc._worker_busy_loop()

    worker_proc._return_result.assert_called_once_with(
        {"status": "error", "error": "model execution failed", "wave_id": None}
    )
    assert original.__traceback__ is None
    gc_collect.assert_called_once_with()
    mock_platform.empty_cache.assert_called_once_with()


def test_busy_loop_preserves_order_when_overflow_payload_is_delayed(monkeypatch):
    """A consumed overflow marker must not be abandoned on a socket timeout."""
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        local_reader_ranks=[0],
        max_chunk_bytes=1024,
        max_chunks=4,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()

    overflow_send_started = threading.Event()
    release_overflow_send = threading.Event()
    overflow_poll_started = threading.Event()
    writer.local_socket = _DelayedMultipartSocket(
        writer.local_socket,
        overflow_send_started,
        release_overflow_send,
    )
    reader.local_socket = _ObservedPollSocket(reader.local_socket, overflow_poll_started)

    recording_worker = _RecordingWorker()
    worker_proc = WorkerProc.__new__(WorkerProc)
    worker_proc.gpu_id = 0
    worker_proc.mq = reader
    worker_proc.worker = recording_worker
    worker_proc.result_mq = None
    worker_proc.wake_event = None
    worker_proc._running = True

    # Make the pre-fix implementation fail quickly: it read this module-level
    # value directly from _worker_busy_loop. The fixed implementation ignores it
    # and delegates to recv_message(), which waits indefinitely.
    monkeypatch.setattr(
        diffusion_worker_module,
        "_BROADCAST_DEQUEUE_TIMEOUT_S",
        0.02,
        raising=False,
    )

    overflow_message = {
        "type": "rpc",
        "method": "record",
        "args": ("overflow-1",),
        "padding": b"x" * 2048,
    }
    inline_message = {
        "type": "rpc",
        "method": "record",
        "args": ("inline-2",),
    }
    shutdown_message = {"type": "shutdown"}
    thread_errors: list[BaseException] = []

    def send_messages():
        try:
            writer.enqueue(overflow_message)
            writer.enqueue(inline_message)
            writer.enqueue(shutdown_message)
        except BaseException as exc:
            thread_errors.append(exc)

    def run_busy_loop():
        try:
            worker_proc._worker_busy_loop()
        except BaseException as exc:
            thread_errors.append(exc)

    writer_thread = threading.Thread(target=send_messages, daemon=True)
    worker_thread = threading.Thread(target=run_busy_loop, daemon=True)

    try:
        writer_thread.start()
        worker_thread.start()

        assert overflow_send_started.wait(timeout=2)
        assert overflow_poll_started.wait(timeout=2)

        # Keep the multipart unavailable beyond the pre-fix 20 ms timeout.
        time.sleep(0.1)
        release_overflow_send.set()

        writer_thread.join(timeout=2)
        worker_thread.join(timeout=2)

        assert not writer_thread.is_alive()
        assert not worker_thread.is_alive()
        assert thread_errors == []
        assert recording_worker.calls == ["overflow-1", "inline-2"]
    finally:
        release_overflow_send.set()
        worker_proc._running = False
        reader.shutdown()
        writer.shutdown()
        writer_thread.join(timeout=1)
        worker_thread.join(timeout=1)


def test_busy_loop_reads_pipeline_and_broadcast_commands_without_polling() -> None:
    broadcast_writer = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
    pipeline_writer = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
    broadcast_reader = MessageQueue.create_from_handle(broadcast_writer.export_handle(), rank=0)
    pipeline_reader = MessageQueue.create_from_handle(pipeline_writer.export_handle(), rank=0)
    for writer, reader in ((broadcast_writer, broadcast_reader), (pipeline_writer, pipeline_reader)):
        writer.wait_until_ready()
        reader.wait_until_ready()

    worker_proc = WorkerProc.__new__(WorkerProc)
    worker_proc.gpu_id = 0
    worker_proc.mq = broadcast_reader
    worker_proc.pipeline_command_mq = pipeline_reader
    worker_proc.worker = _RecordingWorker()
    worker_proc.result_mq = None
    worker_proc.wake_event = None
    worker_proc._running = True
    worker_proc._control_messages = diffusion_worker_module.queue.PriorityQueue(maxsize=2)
    worker_proc._control_message_sequence = diffusion_worker_module.itertools.count()
    worker_proc._control_reader_threads = []
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            worker_proc._worker_busy_loop()
        except BaseException as exc:
            errors.append(exc)

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    try:
        worker_thread.start()
        pipeline_writer.enqueue({"type": "rpc", "method": "record", "args": ("pipeline",)})
        broadcast_writer.enqueue({"type": "rpc", "method": "record", "args": ("broadcast",)})
        deadline = time.monotonic() + 2
        while len(worker_proc.worker.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        broadcast_writer.enqueue({"type": "shutdown"})
        worker_thread.join(timeout=2)

        assert not worker_thread.is_alive()
        assert errors == []
        assert set(worker_proc.worker.calls) == {"pipeline", "broadcast"}
    finally:
        worker_proc._running = False
        for reader in (broadcast_reader, pipeline_reader):
            reader.shutdown()
        for writer in (broadcast_writer, pipeline_writer):
            writer.shutdown()
        worker_thread.join(timeout=1)
        for reader_thread in worker_proc._control_reader_threads:
            reader_thread.join(timeout=1)
