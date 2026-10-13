# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading

import pytest

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_forward_context_isolated_between_worker_threads() -> None:
    barrier = threading.Barrier(2)
    observed: dict[str, int | None] = {}

    def read_context(name: str, step: int) -> None:
        assert not is_forward_context_available()
        with override_forward_context(ForwardContext(denoise_step_idx=step)):
            barrier.wait(timeout=5.0)
            observed[name] = get_forward_context().denoise_step_idx

    threads = [
        threading.Thread(target=read_context, args=("left", 3)),
        threading.Thread(target=read_context, args=("right", 8)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert observed == {"left": 3, "right": 8}
    assert not is_forward_context_available()
