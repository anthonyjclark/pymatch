from __future__ import annotations

from collections.abc import Callable, Sequence
from math import exp, log, tanh
from typing import cast

type NestedSequence[T] = T | Sequence[NestedSequence[T]]

# TODO: consider numbers.Real and numbers.Integral
type Scalar = int | float
type NestedArray = Scalar | Sequence[NestedArray[Scalar]]
type NestedInts = int | Sequence[NestedInts[int]]
type Shape = tuple[int, ...]
type Index = NestedInts | NDArray
type BinaryOp[T] = Callable[[T, T], T]


def _shape_cast(*shape: int | Shape) -> Shape:
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return tuple(shape[0])
    return cast(Shape, shape)


def _shape_to_size(shape: Shape) -> int:
    size = 1
    for d in shape:
        size *= d
    return size


class NDArray:
    data: list[Scalar]
    shape: Shape
    strides: Shape
    dtype: type[Scalar]

    def __init__(
        self, data: NestedArray | NDArray, shape: int | Shape | None = None, dtype: type[Scalar] = float
    ) -> None:
        """
        Initialize an NDArray.

        Args:
            data: The data to initialize the array with (NDArray, scalar, list, or tuple).
            shape: The shape of the array. If None, inferred from data.
            dtype: The data type of the array elements.
        """

        # TODO: enforce dtype on data
        self.dtype = dtype

        if isinstance(shape, int):
            shape = (shape,)
        if shape is not None:
            shape = tuple(shape if isinstance(shape, (list, tuple)) else (shape,))

        if isinstance(data, float):
            self.data = [float(data)]
            self.shape = () if shape is None else shape

        elif isinstance(data, (list, tuple)):
            flat_data, detected_shape = self._flatten(data)
            self.data = [float(x) for x in flat_data]
            self.shape = tuple(shape) if shape is not None else detected_shape

        elif isinstance(data, NDArray):
            self.data = list(data.data)
            self.shape = data.shape if shape is None else shape

        else:
            raise TypeError(f"Unsupported data type for NDArray: {type(data)}")

        n = len(self.data)
        shape_n = _shape_to_size(self.shape)
        if (not self.shape and n != 1) or (shape_n != n):
            raise ValueError(f"Data size {n} does not match size {shape_n} for shape {self.shape}")

        self.strides = self._calc_strides(self.shape)

    @staticmethod
    def _flatten(seq: NestedArray) -> tuple[list[Scalar], Shape]:
        if isinstance(seq, (int, float)):
            return [seq], ()

        if isinstance(seq, (list, tuple)) and len(seq) == 0:
            return [], (0,)

        flat: list[Scalar] = []
        elem_shapes: list[Shape] = []

        assert isinstance(seq, (list, tuple))
        for item in seq:
            sub_flat, sub_shape = NDArray._flatten(item)
            flat.extend(sub_flat)
            elem_shapes.append(sub_shape)

        dim0 = len(seq)
        sub_shape = elem_shapes[0]
        for s in elem_shapes:
            if s != sub_shape:
                raise ValueError(f"Inconsistent shape in nested list: {s} vs {sub_shape}")

        return flat, (dim0,) + sub_shape

    """
    TODO: Implement unflatten
    """

    @staticmethod
    def _calc_strides(shape: Shape) -> Shape:
        if not shape:
            return ()

        strides = [1] * len(shape)

        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        return tuple(strides)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return len(self.data)

    def copy(self) -> NDArray:
        return NDArray(self)

    def fill(self, val: float | int) -> None:
        val_float = float(val)
        for i in range(len(self.data)):
            self.data[i] = val_float

    def item(self) -> float:
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return float(self.data[0])

    def tolist(self) -> NestedArray:
        if self.size == 1:
            return list(self.data)

        def _unflatten(data: list[Scalar], shape: Shape) -> NestedArray:
            if len(shape) == 1:
                return list(data)

            step = 1
            for d in shape[1:]:
                step *= d

            return [_unflatten(data[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]

        return _unflatten(self.data, self.shape)

    def reshape(self, *shape: int | Shape) -> NDArray:
        shape_tup = _shape_cast(*shape)

        if -1 in shape_tup:
            neg_idx = shape_tup.index(-1)
            known_prod = 1
            for i, d in enumerate(shape_tup):
                if i != neg_idx:
                    known_prod *= d
            calc_dim = self.size // known_prod
            shape_tup = shape_tup[:neg_idx] + (calc_dim,) + shape_tup[neg_idx + 1 :]

        new_size = 1
        for d in shape_tup:
            new_size *= d

        if new_size != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {shape_tup}")

        return NDArray(list(self.data), shape=shape_tup)

    @property
    def T(self) -> NDArray:
        return self.transpose()

    def transpose(self) -> NDArray:
        if self.ndim < 2:
            return self.copy()

        if self.ndim == 2:
            rows, cols = self.shape
            out_data = [0.0] * (rows * cols)
            for r in range(rows):
                for c in range(cols):
                    out_data[c * rows + r] = self.data[r * cols + c]
            return NDArray(out_data, shape=(cols, rows))

        new_shape = tuple(reversed(self.shape))
        out_data = [0.0] * self.size
        for i in range(self.size):
            coords = self._unravel_index(i, self.shape)
            rev_coords = tuple(reversed(coords))
            new_idx = self._ravel_index(rev_coords, new_shape)
            out_data[new_idx] = self.data[i]

        return NDArray(out_data, shape=new_shape)

    def _unravel_index(self, index: int, shape: Shape) -> Shape:
        if not shape:
            return ()

        coords: list[int] = []
        for d in reversed(shape):
            coords.append(index % d if d != 0 else 0)
            index //= d if d != 0 else 1

        return tuple(reversed(coords))

    def _ravel_index(self, coords: Shape, shape: Shape) -> int:
        if not shape:
            return 0

        idx = 0
        stride = 1
        for c, d in zip(reversed(coords), reversed(shape)):
            idx += c * stride
            stride *= d

        return idx

    @staticmethod
    def _broadcast_shapes(s1: Shape, s2: Shape) -> tuple[Shape, Shape, Shape]:
        l1, l2 = len(s1), len(s2)
        max_len = max(l1, l2)
        s1_pad = (1,) * (max_len - l1) + s1
        s2_pad = (1,) * (max_len - l2) + s2

        out_shape: list[int] = []
        for d1, d2 in zip(s1_pad, s2_pad):
            if d1 == 1:
                out_shape.append(d2)
            elif d2 == 1:
                out_shape.append(d1)
            elif d1 == d2:
                out_shape.append(d1)
            else:
                raise ValueError(f"Operands could not be broadcast together with shapes {s1} {s2}")

        return tuple(out_shape), s1_pad, s2_pad

    def _binary_op(self, other: float | NDArray, op_fn: BinaryOp[float]) -> NDArray:
        if not isinstance(other, NDArray):
            other = NDArray(other)

        out_shape, s1_pad, s2_pad = NDArray._broadcast_shapes(self.shape, other.shape)
        out_size = 1
        for d in out_shape:
            out_size *= d

        out_data = [0.0] * out_size
        strides_a = NDArray._calc_strides(s1_pad)
        strides_b = NDArray._calc_strides(s2_pad)

        for i in range(out_size):
            coords = self._unravel_index(i, out_shape)
            idx_a = sum((c if dim > 1 else 0) * st for c, dim, st in zip(coords, s1_pad, strides_a))
            idx_b = sum((c if dim > 1 else 0) * st for c, dim, st in zip(coords, s2_pad, strides_b))
            out_data[i] = op_fn(self.data[idx_a], other.data[idx_b])

        return NDArray(out_data, shape=out_shape)

    # TODO: consider using a switch above, or importing operators instead of using lambdas

    def __add__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: float | NDArray) -> NDArray:
        return NDArray(other).__add__(self)

    def __sub__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: float | NDArray) -> NDArray:
        return NDArray(other).__sub__(self)

    def __mul__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: float | NDArray) -> NDArray:
        return NDArray(other).__truediv__(self)

    def __pow__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: a**b)

    def __rpow__(self, other: float | NDArray) -> NDArray:
        return NDArray(other).__pow__(self)

    def __neg__(self) -> NDArray:
        return NDArray([-x for x in self.data], shape=self.shape)

    def __gt__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: 1.0 if a > b else 0.0)

    def __lt__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: 1.0 if a < b else 0.0)

    def __ge__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: 1.0 if a >= b else 0.0)

    def __le__(self, other: float | NDArray) -> NDArray:
        return self._binary_op(other, lambda a, b: 1.0 if a <= b else 0.0)

    def __eq__(self, other: float | NDArray) -> NDArray:  # type: ignore
        return self._binary_op(other, lambda a, b: 1.0 if a == b else 0.0)

    def __matmul__(self, other: float | NDArray) -> NDArray:
        if not isinstance(other, NDArray):
            other = NDArray(other)

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(f"Matmul requires 2D arrays, got ndim {self.ndim} and {other.ndim}")

        M, K = self.shape
        K_b, N = other.shape
        if K != K_b:
            raise ValueError(f"Cannot matmul shapes {self.shape} and {other.shape}")

        c_data = [0.0] * (M * N)
        a_data = self.data
        b_data = other.data
        for i in range(M):
            r_off = i * K
            out_off = i * N
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a_data[r_off + k] * b_data[k * N + j]
                c_data[out_off + j] = s
        return NDArray(c_data, shape=(M, N))

    def relu(self) -> NDArray:
        return NDArray([x if x > 0.0 else 0.0 for x in self.data], shape=self.shape)

    def sigmoid(self) -> NDArray:
        return NDArray([1.0 / (1.0 + exp(-max(-50.0, min(50.0, x)))) for x in self.data], shape=self.shape)

    def tanh(self) -> NDArray:
        return NDArray([tanh(x) for x in self.data], shape=self.shape)

    def exp(self) -> NDArray:
        return NDArray([exp(x) for x in self.data], shape=self.shape)

    def log(self) -> NDArray:
        return NDArray([log(x) for x in self.data], shape=self.shape)

    def sum(self, axis: int | Shape | None = None, keepdims: bool = False) -> NDArray:
        if axis is None:
            total = sum(self.data)
            return NDArray([total], shape=() if not keepdims else (1,) * self.ndim)

        if isinstance(axis, int):
            axis = (axis,)

        axis_normalized = tuple(a % self.ndim for a in axis)

        out_shape: list[int] = []
        for i, d in enumerate(self.shape):
            if i in axis_normalized:
                if keepdims:
                    out_shape.append(1)
            else:
                out_shape.append(d)
        final_out_shape = tuple(out_shape)
        out_size = 1
        for d in final_out_shape:
            out_size *= d
        if out_size == 0:
            out_size = 1

        out_data = [0.0] * out_size

        for i in range(self.size):
            coords = self._unravel_index(i, self.shape)
            out_coords: list[int] = []
            for ax, c in enumerate(coords):
                if ax not in axis_normalized:
                    out_coords.append(c)
                elif keepdims:
                    out_coords.append(0)
            final_out_coords = tuple(out_coords)
            out_idx = self._ravel_index(final_out_coords, final_out_shape) if final_out_coords else 0
            out_data[out_idx] += self.data[i]

        return NDArray(out_data, shape=final_out_shape)

    def mean(self, axis: int | Shape | None = None, keepdims: bool = False) -> NDArray:
        s = self.sum(axis=axis, keepdims=keepdims)
        num_elem = self.size if axis is None else 1
        if axis is not None:
            axis = (axis,) if isinstance(axis, int) else axis
            for a in axis:
                num_elem *= self.shape[a % self.ndim]
        return s / float(num_elem)

    def __getitem__(self, idx: Index) -> float | NDArray:
        # TODO: need an int NDArray for indexing
        if isinstance(idx, int):
            idx = (idx,)

        elif isinstance(idx, Sequence):
            idx = idx

        elif isinstance(idx, NDArray):
            assert idx.dtype == int, "Indexing NDArray must be of integer type"
            flat_mask = idx.data
            out_data = [self.data[i] for i, m in enumerate(flat_mask) if m != 0.0]
            return NDArray(out_data, shape=(len(out_data),))

        if len(idx) == 1 and isinstance(idx[0], int) and self.ndim == 1:
            return self.data[idx[0]]

        elif len(idx) == 2 and isinstance(idx[0], int) and isinstance(idx[1], int) and self.ndim == 2:
            r: int = idx[0]
            c: int = idx[1]
            return self.data[r * self.shape[1] + c]

        elif len(idx) == 1 and isinstance(idx[0], int) and self.ndim == 2:
            r = idx[0]
            row_data = self.data[r * self.shape[1] : (r + 1) * self.shape[1]]
            return NDArray(row_data, shape=(self.shape[1],))

        # TODO: replace, this case is too inefficient
        res = self.tolist()

        for i in idx:
            res = res[i]  # type: ignore

        return NDArray(res)

    # TODO: support nested items for values?
    def __setitem__(self, idx: Index, value: NDArray) -> None:
        val_float = float(value) if isinstance(value, (int, float)) else float(value.data[0])
        if isinstance(idx, tuple) and len(idx) == 2 and self.ndim == 2:
            r: int = idx[0]  # type: ignore
            c: int = idx[1]  # type: ignore
            self.data[r * self.shape[1] + c] = val_float
        elif isinstance(idx, int) and self.ndim == 1:
            self.data[idx] = val_float
        else:
            raise NotImplementedError("Setitem currently supported for 1D/2D indices")

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f"{self.data[0]:.4f}"
        return f"{self.tolist()}"
