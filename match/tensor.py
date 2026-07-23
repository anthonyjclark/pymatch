import math
import random
from ._ndarray import NDArray


def manual_seed(seed: int):
    random.seed(seed)


def _unbroadcast(grad: NDArray, target_shape: tuple) -> NDArray:
    if grad.shape == target_shape:
        return grad
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)), keepdims=False)

    axes_to_sum = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_sum.append(i)
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    if grad.shape != target_shape:
        grad = grad.reshape(*target_shape)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Initialize a Tensor.

        Args:
            data: The data to initialize the tensor with.
            requires_grad: Whether the tensor requires gradient computation.
        """
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, NDArray):
            self.data = data.copy()
        else:
            self.data = NDArray(data)

        self.requires_grad = requires_grad
        self.grad = None
        if requires_grad:
            self.grad = NDArray([0.0] * self.data.size, shape=self.data.shape)

        self.creator = None
        self._backward = lambda: None
        self.op_name = ""

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        return self.data.item()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, gradient=None):
        if not self.requires_grad:
            return

        if gradient is None:
            if self.shape == () or self.data.size == 1:
                gradient = NDArray([1.0], shape=self.shape)
            else:
                raise RuntimeError("Grad can be implicitly created only for scalar outputs")
        elif not isinstance(gradient, NDArray):
            gradient = NDArray(gradient, shape=self.shape)

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.creator is not None:
                    for child in v.creator:
                        build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.creator = (self, other)
        out.op_name = "+"

        def _backward():
            if self.requires_grad:
                grad_self = _unbroadcast(out.grad, self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = _unbroadcast(out.grad, other.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return Tensor(other).__add__(self)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.creator = (self, other)
        out.op_name = "-"

        def _backward():
            if self.requires_grad:
                grad_self = _unbroadcast(out.grad, self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = _unbroadcast(-out.grad, other.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.creator = (self, other)
        out.op_name = "*"

        def _backward():
            if self.requires_grad:
                grad_self = _unbroadcast(out.grad * other.data, self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = _unbroadcast(out.grad * self.data, other.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return Tensor(other).__mul__(self)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.creator = (self, other)
        out.op_name = "/"

        def _backward():
            if self.requires_grad:
                grad_self = _unbroadcast(out.grad / other.data, self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other).__truediv__(self)

    def __pow__(self, other):
        is_tensor = isinstance(other, Tensor)
        other_data = other.data if is_tensor else other
        out = Tensor(self.data ** (other_data.data if is_tensor else other_data), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "**"

        def _backward():
            if self.requires_grad:
                p = other_data.data if is_tensor else other_data
                grad_self = _unbroadcast(out.grad * (p * (self.data ** (p - 1.0))), self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def __rpow__(self, other):
        return Tensor(other).__pow__(self)

    def __neg__(self):
        return self * (-1.0)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.creator = (self, other)
        out.op_name = "@"

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(self.data.relu(), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "relu"

        def _backward():
            if self.requires_grad:
                mask = NDArray([1.0 if x > 0.0 else 0.0 for x in self.data.data], shape=self.shape)
                grad_self = out.grad * mask
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(self.data.sigmoid(), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "sigmoid"

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * out.data * (1.0 - out.data)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(self.data.tanh(), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "tanh"

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * (1.0 - (out.data ** 2))
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "sum"

        def _backward():
            if self.requires_grad:
                expanded_grad = out.grad
                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis, int) else axis
                    axes = tuple(a % self.ndim for a in axes)
                    new_shape = list(self.shape)
                    for a in axes:
                        new_shape[a] = 1
                    expanded_grad = expanded_grad.reshape(*new_shape)

                ones_like = NDArray([1.0] * self.data.size, shape=self.shape)
                grad_self = ones_like * expanded_grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        num_elem = self.data.size if axis is None else 1
        if axis is not None:
            axes = (axis,) if isinstance(axis, int) else axis
            for a in axes:
                num_elem *= self.shape[a % self.ndim]
        return s / float(num_elem)

    def reshape(self, *new_shape):
        out = Tensor(self.data.reshape(*new_shape), requires_grad=self.requires_grad)
        out.creator = (self,)
        out.op_name = "reshape"

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.reshape(*self.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    return Tensor(NDArray([0.0] * size, shape=shape), requires_grad=requires_grad)


def ones(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    return Tensor(NDArray([1.0] * size, shape=shape), requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(NDArray(data, shape=shape), requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    data = [random.random() for _ in range(size)]
    return Tensor(NDArray(data, shape=shape), requires_grad=requires_grad)
