from __future__ import annotations

import array
import gzip
import os
import struct
import sys

from .._ndarray import NDArray
from ..tensor import Tensor
from ..utils.data import TensorDataset

IN_PYODIDE = "pyodide" in sys.modules or sys.platform == "emscripten"


def _parse_bin_file(bin_path: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    with open(bin_path, "rb") as f:
        raw_data = f.read()

    if raw_data[:2] == b"\x1f\x8b":
        data = gzip.decompress(raw_data)
    else:
        data = raw_data

    magic, num_train, num_valid, num_features = struct.unpack("<IIII", data[:16])
    if magic != 0x4D4E4953:
        raise ValueError("Invalid binary blob magic header")

    offset = 16
    tx_bytes_len = num_train * num_features * 4
    tx_bytes = data[offset : offset + tx_bytes_len]
    offset += tx_bytes_len

    ty_bytes_len = num_train * 4
    ty_bytes = data[offset : offset + ty_bytes_len]
    offset += ty_bytes_len

    vx_bytes_len = num_valid * num_features * 4
    vx_bytes = data[offset : offset + vx_bytes_len]
    offset += vx_bytes_len

    vy_bytes = data[offset : offset + num_valid * 4]

    train_x_arr = array.array("f")
    train_x_arr.frombytes(tx_bytes)

    train_y_arr = array.array("f")
    train_y_arr.frombytes(ty_bytes)

    valid_x_arr = array.array("f")
    valid_x_arr.frombytes(vx_bytes)

    valid_y_arr = array.array("f")
    valid_y_arr.frombytes(vy_bytes)

    return (
        NDArray._create_flat(train_x_arr, shape=(num_train, num_features)),
        NDArray._create_flat(train_y_arr, shape=(num_train,)),
        NDArray._create_flat(valid_x_arr, shape=(num_valid, num_features)),
        NDArray._create_flat(valid_y_arr, shape=(num_valid,)),
    )


def load_mnist_dataset(
    data_dir: str | None = None, download: bool = True, flatten: bool = True
) -> tuple[TensorDataset, TensorDataset]:
    """Load MNIST dataset partitions as PyMatch TensorDatasets from a preprocessed binary blob (mnist.bin).

    Args:
        data_dir (str | None): Directory path containing dataset binary blob (mnist.bin). If None, uses package resource.
        download (bool): Ignored. Dataset binary blob is pre-baked locally or in Pyodide VFS.
        flatten (bool): If True, returns 2D tensors of shape (N, 784), otherwise (N, 1, 28, 28).

    Returns:
        tuple[TensorDataset, TensorDataset]: (train_dataset, valid_dataset)

    Raises:
        FileNotFoundError: If mnist.bin is not found.
    """
    if IN_PYODIDE or "pyodide" in sys.modules:
        bin_path = "/data/mnist.bin"
    elif data_dir is not None:
        bin_path = os.path.join(data_dir, "mnist.bin")
    else:
        bin_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "resources", "mnist.bin")
        )

    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Could not find dataset blob at '{bin_path}'. "
            "Please run 'npm run build:mnist' to generate 'match/resources/mnist.bin'."
        )

    nd_tx, nd_ty, nd_vx, nd_vy = _parse_bin_file(bin_path)

    if not flatten:
        nd_tx = nd_tx.reshape(nd_tx.shape[0], 1, 28, 28)
        nd_vx = nd_vx.reshape(nd_vx.shape[0], 1, 28, 28)

    X_train = Tensor._create(nd_tx)
    y_train = Tensor._create(nd_ty)
    X_valid = Tensor._create(nd_vx)
    y_valid = Tensor._create(nd_vy)

    return TensorDataset(X_train, y_train), TensorDataset(X_valid, y_valid)


def get_binary_mnist_one_batch(
    data_dir: str | None = None,
    classA: int = 1,
    classB: int = 7,
    flatten: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Filter MNIST to binary classes A and B."""
    train_ds, valid_ds = load_mnist_dataset(data_dir, flatten=flatten)
    X_train, y_train = train_ds.tensors
    X_valid, y_valid = valid_ds.tensors

    def filter_partition(x_tensor: Tensor, y_tensor: Tensor):
        labels = y_tensor.data.data
        mask = [val == classA or val == classB for val in labels]
        filtered_y = [1.0 if val == classB else 0.0 for val, keep in zip(labels, mask) if keep]
        n_samples = len(filtered_y)
        num_features = 784

        filtered_x = []
        for i, keep in enumerate(mask):
            if keep:
                start = i * num_features
                filtered_x.extend(x_tensor.data.data[start : start + num_features])

        shape_x = (n_samples, 784) if flatten else (n_samples, 1, 28, 28)
        return (
            Tensor._create(NDArray(filtered_x, shape=shape_x)),
            Tensor._create(NDArray(filtered_y, shape=(n_samples,))),
        )

    tr_x, tr_y = filter_partition(X_train, y_train)
    val_x, val_y = filter_partition(X_valid, y_valid)

    return tr_x, tr_y, val_x, val_y
