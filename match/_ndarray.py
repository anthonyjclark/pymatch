class NDArray:
    def __init__(self, data, shape=None):
        """
        Initialize an NDArray.

        Args:
            data: The data to initialize the array with.
            shape: The shape of the array. If None, the shape will be inferred from the data.
        """

        # Copying an existing NDArray
        if isinstance(data, NDArray):
            self.data = list(data.data)
            self.shape = data.shape
            self.strides = data.strides

        # Initializing from a single number (int or float)
        elif isinstance(data, (int, float)):
            self.data = [float(data)]
            self.shape = () if shape is None else tuple(shape)
            self.strides = self._calc_strides(self.shape)

        else:
            raise TypeError(f"Unsupported data type for NDArray: {type(data)}")

    @staticmethod
    def _calc_strides(shape):
        "Calculate the strides for a given shape."
        if not shape:
            return ()

        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        return tuple(strides)

    @property
    def ndim(self):
        "Return the number of dimensions of the array."
        return len(self.shape)

    @property
    def size(self):
        "Return the total number of elements in the array."
        if not self.shape:
            return 1

        res = 1
        for d in self.shape:
            res *= d
        return res
