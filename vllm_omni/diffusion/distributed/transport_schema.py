"""Shared tensor-dictionary metadata schema for diffusion transports."""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import torch

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any], prefix: str = ""
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Return flattened metadata and tensor leaves for a nested payload."""
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if "%" in key:
            raise ValueError("tensor payload keys cannot contain '%' metadata separators")
        flattened_key = prefix + key
        if isinstance(value, torch.Tensor):
            metadata_list.append((flattened_key, TensorMetadata(value.device.type, value.dtype, value.size())))
            tensor_list.append(value)
        elif isinstance(value, dict):
            if not value:
                metadata_list.append((flattened_key, value))
            inner_metadata, inner_tensors = split_tensor_dict(value, flattened_key + "%")
            metadata_list.extend(inner_metadata)
            tensor_list.extend(inner_tensors)
        else:
            metadata_list.append((flattened_key, value))
    return metadata_list, tensor_list


def update_nested_dict(nested_dict: dict[str, Any], flattened_key: str, value: Any) -> None:
    """Insert one flattened metadata entry into its nested payload shape."""
    current = nested_dict
    key_splits = flattened_key.split("%")
    for key in key_splits[:-1]:
        current = current.setdefault(key, {})
    current[key_splits[-1]] = value
