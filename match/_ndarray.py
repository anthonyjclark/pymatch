import math


class NDArray:
    def __init__(self, data, shape=None):
        """
        Initialize an NDArray.

        Args:
            data: The data to initialize the array with (NDArray, scalar, list, or tuple).
            shape: The shape of the array. If None, inferred from data.
        """
        if isinstance(data, NDArray):
            self.data = list(data.data)
            self.shape = data.shape if shape is None else tuple(shape)
        elif isinstance(data, (int, float)):
            self.data = [float(data)]
            self.shape = () if shape is None else tuple(shape)
        elif isinstance(data, (list, tuple)):
            flat_data, detected_shape = self._flatten(data)
            self.data = [float(x) for x in flat_data]
            self.shape = tuple(shape) if shape is not None else detected_shape
        else:
            if hasattr(data, "tolist"):
                data = data.tolist()
                flat_data, detected_shape = self._flatten(data)
                self.data = [float(x) for x in flat_data]
                self.shape = tuple(shape) if shape is not None else detected_shape
            else:
                raise TypeError(f"Unsupported data type for NDArray: {type(data)}")

        self.strides = self._calc_strides(self.shape)

    @staticmethod
    def _flatten(seq):
        if not isinstance(seq, (list, tuple)):
            return [seq], ()
        if len(seq) == 0:
            return [], (0,)

        flat = []
        elem_shapes = []
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

    @staticmethod
    def _calc_strides(shape):
        if not shape:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        if not self.shape:
            return 1
        res = 1
        for d in self.shape:
            res *= d
        return res

    def copy(self):
        return NDArray(list(self.data), shape=self.shape)

    def fill(self, val):
        val = float(val)
        for i in range(len(self.data)):
            self.data[i] = val

    def item(self):
        if self.size == 1:
            return float(self.data[0])
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    def tolist(self):
        if self.ndim == 0:
            return self.data[0]

        def _unflatten(data, shape):
            if len(shape) == 1:
                return data[: shape[0]]
            step = 1
            for d in shape[1:]:
                step *= d
            return [_unflatten(data[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]

        return _unflatten(self.data, self.shape)

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)

        if -1 in new_shape:
            neg_idx = new_shape.index(-1)
            known_prod = 1
            for i, d in enumerate(new_shape):
                if i != neg_idx:
                    known_prod *= d
            calc_dim = self.size // known_prod
            new_shape = new_shape[:neg_idx] + (calc_dim,) + new_shape[neg_idx + 1 :]

        new_size = 1
        for d in new_shape:
            new_size *= d
        if new_size != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")

        return NDArray(list(self.data), shape=new_shape)

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
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

    def _unravel_index(self, index, shape):
        if not shape:
            return ()
        coords = []
        for d in reversed(shape):
            coords.append(index % d if d != 0 else 0)
            index //= d if d != 0 else 1
        return tuple(reversed(coords))

    def _ravel_index(self, coords, shape):
        if not shape:
            return 0
        idx = 0
        stride = 1
        for c, d in zip(reversed(coords), reversed(shape)):
            idx += c * stride
            stride *= d
        return idx

    @staticmethod
    def _broadcast_shapes(s1, s2):
        l1, l2 = len(s1), len(s2)
        max_len = max(l1, l2)
        s1_pad = (1,) * (max_len - l1) + s1
        s2_pad = (1,) * (max_len - l2) + s2

        out_shape = []
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

    def _binary_op(self, other, op_fn):
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

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return NDArray(other).__add__(self)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return NDArray(other).__sub__(self)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return NDArray(other).__truediv__(self)

    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a**b)

    def __rpow__(self, other):
        return NDArray(other).__pow__(self)

    def __neg__(self):
        return NDArray([-x for x in self.data], shape=self.shape)

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: 1.0 if a > b else 0.0)

    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: 1.0 if a < b else 0.0)

    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: 1.0 if a >= b else 0.0)

    def __le__(self, other):
        return self._binary_op(other, lambda a, b: 1.0 if a <= b else 0.0)

    def __eq__(self, other):
        return self._binary_op(other, lambda a, b: 1.0 if a == b else 0.0)

    def __matmul__(self, other):
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

    def relu(self):
        return NDArray([x if x > 0.0 else 0.0 for x in self.data], shape=self.shape)

    def sigmoid(self):
        return NDArray(
            [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x)))) for x in self.data], shape=self.shape
        )

    def tanh(self):
        return NDArray([math.tanh(x) for x in self.data], shape=self.shape)

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            total = sum(self.data)
            return NDArray([total], shape=() if not keepdims else (1,) * self.ndim)

        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(a % self.ndim for a in axis)

        out_shape = []
        for i, d in enumerate(self.shape):
            if i in axis:
                if keepdims:
                    out_shape.append(1)
            else:
                out_shape.append(d)
        out_shape = tuple(out_shape)
        out_size = 1
        for d in out_shape:
            out_size *= d
        if out_size == 0:
            out_size = 1

        out_data = [0.0] * out_size

        for i in range(self.size):
            coords = self._unravel_index(i, self.shape)
            out_coords = []
            for ax, c in enumerate(coords):
                if ax not in axis:
                    out_coords.append(c)
                elif keepdims:
                    out_coords.append(0)
            out_coords = tuple(out_coords)
            out_idx = self._ravel_index(out_coords, out_shape) if out_coords else 0
            out_data[out_idx] += self.data[i]

        return NDArray(out_data, shape=out_shape)

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        num_elem = self.size if axis is None else 1
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for a in axis:
                num_elem *= self.shape[a % self.ndim]
        return s / float(num_elem)

    def __getitem__(self, idx):
        if isinstance(idx, NDArray):
            flat_mask = idx.data
            out_data = [self.data[i] for i, m in enumerate(flat_mask) if m != 0.0]
            return NDArray(out_data, shape=(len(out_data),))

        if isinstance(idx, int):
            idx = (idx,)

        if isinstance(idx, tuple):
            if len(idx) == 1 and isinstance(idx[0], int) and self.ndim == 1:
                return self.data[idx[0]]
            if len(idx) == 2 and all(isinstance(x, int) for x in idx) and self.ndim == 2:
                r, c = idx
                return self.data[r * self.shape[1] + c]
            if len(idx) == 1 and isinstance(idx[0], int) and self.ndim == 2:
                r = idx[0]
                row_data = self.data[r * self.shape[1] : (r + 1) * self.shape[1]]
                return NDArray(row_data, shape=(self.shape[1],))

        res = self.tolist()
        if isinstance(idx, tuple):
            for i in idx:
                res = res[i]
        else:
            res = res[idx]
        return NDArray(res)

    def __setitem__(self, idx, value):
        val_float = float(value) if isinstance(value, (int, float)) else float(value.data[0])
        if isinstance(idx, tuple) and len(idx) == 2 and self.ndim == 2:
            r, c = idx
            self.data[r * self.shape[1] + c] = val_float
        elif isinstance(idx, int) and self.ndim == 1:
            self.data[idx] = val_float
        else:
            raise NotImplementedError("Setitem currently supported for 1D/2D indices")

    def __repr__(self):
        if self.ndim == 0:
            return f"{self.data[0]:.4f}"
        return f"{self.tolist()}"
