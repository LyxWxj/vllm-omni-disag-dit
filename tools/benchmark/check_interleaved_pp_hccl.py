#!/usr/bin/env python3
"""Probe the HCCL primitives used by retained-state interleaved PP clocks.

Run on one Ascend node with one process per PP rank:

    torchrun --standalone --nproc-per-node=4 \
        tools/benchmark/check_interleaved_pp_hccl.py

The probe creates the same directed edge groups as ``PipelineTickRuntime``.
It deliberately orders each edge's communicator bootstrap through a Gloo
barrier before testing the reverse bootstrap P2P and the forward payload P2P.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _edge_pairs(world_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((*zip(range(world_size - 1), range(1, world_size)), (world_size - 1, 0)))


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("the HCCL PP probe requires at least two ranks")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    try:
        control_group = dist.new_group(list(range(world_size)), backend="gloo")
        edges = _edge_pairs(world_size)
        edge_groups = {
            edge: dist.new_group(sorted(edge), backend="hccl")
            for edge in edges
        }

        for edge_index, (source_rank, destination_rank) in enumerate(edges):
            group = edge_groups[(source_rank, destination_rank)]
            if rank in (source_rank, destination_rank):
                bootstrap = torch.tensor([rank], dtype=torch.int64, device=device)
                dist.all_reduce(bootstrap, group=group)
            dist.barrier(group=control_group)

            if rank == source_rank:
                received = torch.empty(1, dtype=torch.int64, device=device)
                dist.irecv(received, src=destination_rank, group=group, tag=edge_index * 2).wait()
                assert int(received.item()) == destination_rank
            elif rank == destination_rank:
                sent = torch.tensor([rank], dtype=torch.int64, device=device)
                dist.isend(sent, dst=source_rank, group=group, tag=edge_index * 2).wait()
            dist.barrier(group=control_group)

            if rank == destination_rank:
                received = torch.empty(1, dtype=torch.int64, device=device)
                dist.irecv(received, src=source_rank, group=group, tag=edge_index * 2 + 1).wait()
                assert int(received.item()) == source_rank
            elif rank == source_rank:
                sent = torch.tensor([rank], dtype=torch.int64, device=device)
                dist.isend(sent, dst=destination_rank, group=group, tag=edge_index * 2 + 1).wait()
            dist.barrier(group=control_group)

        if rank == 0:
            print(f"PASS: ordered HCCL PP edge probe completed for {world_size} ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
