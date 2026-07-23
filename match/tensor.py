from ._ndarray import NDArray


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Initialize a Tensor.

        Args:
            data: The data to initialize the tensor with.
            requires_grad: Whether the tensor requires gradient computation.
        """
        # Copy an existing Tensor
        if isinstance(data, Tensor):
            self.data = data.data.copy()

        # Copy an existing NDArray
        elif isinstance(data, NDArray):
            self.data = data.copy()

        # Delegate to NDArray for other data types (e.g., list, int, float)
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
