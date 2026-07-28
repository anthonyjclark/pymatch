from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .._ndarray import NDArray
from ..tensor import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    tensors: tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        first_len = len(tensors[0])
        if not all(len(t) == first_len for t in tensors):
            raise ValueError("Size mismatch among tensors in TensorDataset")
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, ...]:
        if len(self.tensors) == 1:
            return self.tensors[0][index]
        return tuple(t[index] for t in self.tensors)


def _stack_tensors(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("Cannot stack an empty sequence of tensors")

    requires_grad = any(t.requires_grad for t in tensors)
    first_shape = tensors[0].shape

    stacked_data: list[float] = []
    for t in tensors:
        stacked_data.extend(t.data.data)

    out_shape = (len(tensors),) + first_shape
    return Tensor._create(NDArray(stacked_data, shape=out_shape), requires_grad=requires_grad)


def _default_collate(batch: list[Any]) -> Any:
    if not batch:
        return None

    first = batch[0]
    if isinstance(first, Tensor):
        return _stack_tensors(batch)
    elif isinstance(first, (tuple, list)):
        num_fields = len(first)
        return tuple(_default_collate([samples[i] for samples in batch]) for i in range(num_fields))
    elif isinstance(first, (int, float)):
        return Tensor._create(NDArray([float(x) for x in batch], shape=(len(batch),)))
    else:
        return batch


class DataLoader:
    dataset: Dataset | Tensor
    batch_size: int
    shuffle: bool
    drop_last: bool

    def __init__(
        self,
        dataset: Dataset | Tensor,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        if isinstance(dataset, Tensor):
            self.dataset = TensorDataset(dataset)
        else:
            self.dataset = dataset

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        num_samples = len(self.dataset)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            samples = [self.dataset[idx] for idx in batch_indices]
            yield _default_collate(samples)
