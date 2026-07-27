from __future__ import annotations

import math

from .tensor import Tensor, randn, tensor


class Module:
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    weight: Tensor
    bias: Tensor | None

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        std_dev = 1.0 / math.sqrt(in_features)
        self.weight = tensor(randn(in_features, out_features) * std_dev, requires_grad=True)
        self.bias = tensor(randn(1, out_features), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class MSELoss(Module):
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        diff = x - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if target.ndim == 1 and target.shape != (x.shape[0],):
            target = target.reshape(1, -1)

        N, C = x.shape

        max_val = x.data.data[0]
        for val in x.data.data:
            max_val = max(max_val, val)
        exp_input = (x - max_val).exp()

        sum_exp = exp_input.sum(axis=1, keepdims=True)
        log_softmax = (x - max_val) - sum_exp.log()

        if target.shape == x.shape:
            loss = -(target * log_softmax).sum() / float(N)
        else:
            target_data = [0.0] * (N * C)
            indices = [int(x) for x in target.data.data]
            for i, idx in enumerate(indices):
                target_data[i * C + idx] = 1.0
            target_tensor = tensor(target_data, requires_grad=False).reshape(N, C)
            loss = -(target_tensor * log_softmax).sum() / float(N)

        return loss
