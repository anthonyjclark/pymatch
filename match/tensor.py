from __future__ import annotations

import random
from collections.abc import Callable

from ._ndarray import NDArray, NestedArray, Scalar, Shape, _shape_cast, _shape_to_size


def manual_seed(seed: int) -> None:
    random.seed(seed)


def tensor(data: Tensor | NestedArray | Scalar, requires_grad: bool = False) -> Tensor:
    return Tensor._create(data, requires_grad=requires_grad)


def zeros(*shape: int | tuple[int, ...], requires_grad: bool = False) -> Tensor:
    shape_tup = _shape_cast(*shape)
    size = _shape_to_size(shape_tup)
    return Tensor._create(NDArray([0.0] * size, shape=shape_tup), requires_grad=requires_grad)


def ones(*shape: Shape, requires_grad: bool = False) -> Tensor:
    shape_tup = _shape_cast(*shape)
    size = _shape_to_size(shape_tup)
    return Tensor._create(NDArray([1.0] * size, shape=shape_tup), requires_grad=requires_grad)


def randn(*shape: Shape, requires_grad: bool = False) -> Tensor:
    shape_tup = _shape_cast(*shape)
    size = _shape_to_size(shape_tup)
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor._create(NDArray(data, shape=shape_tup), requires_grad=requires_grad)


def rand(*shape: Shape, requires_grad: bool = False) -> Tensor:
    shape_tup = _shape_cast(*shape)
    size = _shape_to_size(shape_tup)
    data = [random.random() for _ in range(size)]
    return Tensor._create(NDArray(data, shape=shape_tup), requires_grad=requires_grad)


# TODO: maybe move to _ndarray.py
def _unbroadcast(grad: NDArray, target_shape: Shape) -> NDArray:
    if grad.shape == target_shape:
        return grad

    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)), keepdims=False)

    axes_to_sum: list[int] = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_sum.append(i)
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    if grad.shape != target_shape:
        grad = grad.reshape(*target_shape)
    return grad


class Tensor:
    """
    A tensor is a multi-dimensional array of numbers with support for automatic differentiation. It can be used to perform mathematical operations and track gradients for optimization purposes.

    To construct a Tensor, use the `tensor` function.
    """

    data: NDArray
    requires_grad: bool
    grad: NDArray | None

    _children: tuple[Tensor, ...] | None
    _backward: Callable[[], None]
    _label: str

    @staticmethod
    def _create(data: Tensor | NDArray | NestedArray, requires_grad: bool = False) -> Tensor:
        tensor = Tensor.__new__(Tensor)

        if isinstance(data, Tensor):
            tensor.data = data.data.copy()
        elif isinstance(data, NDArray):
            tensor.data = data.copy()
        else:
            tensor.data = NDArray(data)

        tensor.requires_grad = requires_grad
        tensor.grad = None
        if requires_grad:
            tensor.grad = NDArray([0.0] * tensor.data.size, shape=tensor.data.shape)

        tensor._children = None
        tensor._backward = lambda: None
        tensor._label = ""

        return tensor

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return self.data.item()

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, gradient_in: Tensor | None = None) -> None:
        if not self.requires_grad:
            return

        if gradient_in is None:
            if self.shape == () or self.data.size == 1:
                gradient = NDArray([1.0], shape=self.shape)
            else:
                raise RuntimeError("Grad can be implicitly created only for scalar outputs")
        elif isinstance(gradient_in, Tensor):
            gradient = gradient_in.data

        self.grad = gradient if self.grad is None else self.grad + gradient

        compute_graph: list[Tensor] = []
        visited: set[Tensor] = set()

        def topological_sort(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                if v._children is not None:
                    for child in v._children:
                        topological_sort(child)
                compute_graph.append(v)

        topological_sort(self)

        for node in reversed(compute_graph):
            node._backward()

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor._create(other)
        out_data = self.data + other_tensor.data
        out = Tensor._create(out_data, requires_grad=self.requires_grad or other_tensor.requires_grad)
        out._children = (self, other_tensor)
        out._label = "+"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = _unbroadcast(out.grad, self.shape)
                self.grad = self.grad + g if self.grad is not None else g
            if other_tensor.requires_grad and out.grad:
                g = _unbroadcast(out.grad, other_tensor.shape)
                other_tensor.grad = other_tensor.grad + g if other_tensor.grad is not None else g

        out._backward = _backward
        return out

    def __radd__(self, other: Tensor | Scalar) -> Tensor:
        return Tensor._create(other).__add__(self)

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor._create(other)
        out_data = self.data - other_tensor.data
        out = Tensor._create(out_data, requires_grad=self.requires_grad or other_tensor.requires_grad)
        out._children = (self, other_tensor)
        out._label = "-"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = _unbroadcast(out.grad, self.shape)
                self.grad = self.grad + g if self.grad is not None else g
            if other_tensor.requires_grad and out.grad:
                g = _unbroadcast(-out.grad, other_tensor.shape)
                other_tensor.grad = other_tensor.grad + g if other_tensor.grad is not None else g

        out._backward = _backward
        return out

    def __rsub__(self, other: Tensor | Scalar) -> Tensor:
        return Tensor._create(other).__sub__(self)

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor._create(other)
        out_data = self.data * other_tensor.data
        out = Tensor._create(out_data, requires_grad=self.requires_grad or other_tensor.requires_grad)
        out._children = (self, other_tensor)
        out._label = "*"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = _unbroadcast(out.grad * other_tensor.data, self.shape)
                self.grad = self.grad + g if self.grad is not None else g
            if other_tensor.requires_grad and out.grad:
                g = _unbroadcast(out.grad * self.data, other_tensor.shape)
                other_tensor.grad = other_tensor.grad + g if other_tensor.grad is not None else g

        out._backward = _backward
        return out

    def __rmul__(self, other: Tensor | Scalar) -> Tensor:
        return Tensor._create(other).__mul__(self)

    def __truediv__(self, other: Tensor | Scalar) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor._create(other)
        out_data = self.data / other_tensor.data
        out = Tensor._create(out_data, requires_grad=self.requires_grad or other_tensor.requires_grad)
        out._children = (self, other_tensor)
        out._label = "/"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = _unbroadcast(out.grad / other_tensor.data, self.shape)
                self.grad = self.grad + g if self.grad is not None else g
            if other_tensor.requires_grad and out.grad:
                g = _unbroadcast(-out.grad * self.data / (other_tensor.data**2), other_tensor.shape)
                other_tensor.grad = other_tensor.grad + g if other_tensor.grad is not None else g

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Tensor | Scalar) -> Tensor:
        return Tensor._create(other).__truediv__(self)

    def __pow__(self, other: Tensor | Scalar) -> Tensor:
        p = other.data.data if isinstance(other, Tensor) else other
        out_data = self.data**p  # type: ignore
        out = Tensor._create(out_data, requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "**"

        def _backward() -> None:
            if self.requires_grad and out.grad and isinstance(p, (int, float, NDArray)):
                g = _unbroadcast(out.grad * (p * (self.data ** (p - 1.0))), self.shape)
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def __rpow__(self, other: Tensor | Scalar) -> Tensor:
        return Tensor._create(other).__pow__(self)

    def __neg__(self) -> Tensor:
        return self * (-1.0)

    def __matmul__(self, other: Tensor | Scalar) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor._create(other)
        out_data = self.data @ other_tensor.data
        out = Tensor._create(out_data, requires_grad=self.requires_grad or other_tensor.requires_grad)
        out._children = (self, other_tensor)
        out._label = "@"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = out.grad @ other_tensor.data.T
                self.grad = self.grad + g if self.grad is not None else g
            if other_tensor.requires_grad and out.grad:
                g = self.data.T @ out.grad
                other_tensor.grad = other_tensor.grad + g if other_tensor.grad is not None else g

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor._create(self.data.relu(), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "relu"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                mask = NDArray([1.0 if x > 0.0 else 0.0 for x in self.data.data], shape=self.shape)
                g = out.grad * mask
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        out = Tensor._create(self.data.sigmoid(), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "sigmoid"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = out.grad * out.data * (1.0 - out.data)
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out = Tensor._create(self.data.tanh(), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "tanh"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = out.grad * (1.0 - (out.data**2))
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor._create(self.data.exp(), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "exp"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = out.grad * out.data
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor._create(self.data.log(), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "log"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                g = out.grad / self.data
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def sum(self, axis: int | Shape | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor._create(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "sum"

        def _backward() -> None:
            if self.requires_grad and out.grad:
                expanded_grad = out.grad
                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis, int) else axis
                    axes_normalized = tuple(a % self.ndim for a in axes)
                    new_shape = list(self.shape)
                    for a in axes_normalized:
                        new_shape[a] = 1
                    expanded_grad = expanded_grad.reshape(*new_shape)

                ones_like = NDArray([1.0] * self.data.size, shape=self.shape)
                g = ones_like * expanded_grad
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def mean(self, axis: int | Shape | None = None, keepdims: bool = False) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        num_elem = self.data.size if axis is None else 1
        if axis is not None:
            axes = (axis,) if isinstance(axis, int) else axis
            for a in axes:
                num_elem *= self.shape[a % self.ndim]
        return s / float(num_elem)

    def reshape(self, *new_shape: Shape) -> Tensor:
        out = Tensor._create(self.data.reshape(*new_shape), requires_grad=self.requires_grad)
        out._children = (self,)
        out._label = "reshape"

        def _backward() -> None:
            if self.requires_grad and out.grad is not None:
                g = out.grad.reshape(*self.shape)
                self.grad = self.grad + g if self.grad is not None else g

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
