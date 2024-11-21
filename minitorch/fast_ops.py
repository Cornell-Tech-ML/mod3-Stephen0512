from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A decorator that JIT compiles a function for parallel execution."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check if the tensors are stride-aligned
        if (
            len(out_strides) == len(in_strides)
            and np.array_equal(out_strides, in_strides)
            and np.array_equal(out_shape, in_shape)
        ):
            # Fast path: tensors are stride-aligned, avoid indexing
            for i in prange(out.size):
                out[i] = fn(in_storage[i])
            return

        # Slow path: tensors are not stride-aligned

        # Process each element of the output tensor in parallel
        for i in prange(out.size):
            # Initialize index arrays for input and output tensors
            out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index
            in_index = np.empty(MAX_DIMS, np.int32)  # Input tensor index

            # Convert flat index i to tensor indices for output tensor
            to_index(i, out_shape, out_index)

            # Handle broadcasting between tensors to get input tensor index
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Convert indices to positions in storage
            in_pos = index_to_position(in_index, in_strides)  # Input position
            out_pos = index_to_position(out_index, out_strides)  # Output position

            # Apply function and store result
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Check if the tensors are stride-aligned
        if (
            len(out_strides) == len(a_strides) == len(b_strides)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        ):
            # Fast path: tensors are stride-aligned, avoid indexing
            for i in prange(out.size):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        # Slow path: tensors are not stride-aligned

        # Process each element in the output tensor in parallel
        for i in prange(out.size):
            # Initialize index arrays for input and output tensor indices
            out_index = np.empty(MAX_DIMS, np.int32)  # Output tensor index
            a_index = np.empty(MAX_DIMS, np.int32)  # First input tensor index
            b_index = np.empty(MAX_DIMS, np.int32)  # Second input tensor index

            # Convert flat index i to tensor indices for output tensor
            to_index(i, out_shape, out_index)

            # Handle broadcasting between tensors to get input tensor indices
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Convert indices to positions in storage
            a_pos = index_to_position(a_index, a_strides)  # First input position
            b_pos = index_to_position(b_index, b_strides)  # Second input position
            out_pos = index_to_position(out_index, out_strides)  # Output position

            # Apply function and store result
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Calculate the size of the reduction dimension for the inner loop
        reduce_size = a_shape[reduce_dim]

        # Process each element in the output tensor in parallel
        for i in prange(out.size):
            # Create index buffers for input tensor index
            index = np.empty(
                MAX_DIMS, np.int32
            )  # Tensor index for output first and then for input

            # Convert flat index to output index
            to_index(i, out_shape, index)

            # Convert output index to position in output tensor storage for final output update
            out_pos = index_to_position(index, out_strides)

            # Initialize reduction with first element of the reduction dimension in input tensor
            index[reduce_dim] = 0
            in_pos = index_to_position(
                index, a_strides
            )  # Convert index to position in input tensor storage

            # Initialize accumulated value with the first element of the reduction dimension in input tensor
            accumulated_value = a_storage[in_pos]

            # Inner reduction loop for each element in the reduction dimension (apart from the first one)
            for j in range(1, reduce_size):
                # Update index for next position in reduction dimension
                index[reduce_dim] = j
                in_pos = index_to_position(index, a_strides)

                # Apply reduction function to accumulate result
                accumulated_value = fn(accumulated_value, a_storage[in_pos])

            # Write final accumulated result to output tensor storage
            out[out_pos] = accumulated_value

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Calculate batch stride for tensor a and b
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # a = [[1, 2], [3, 4]] * b = [[5, 6], [7, 8]] = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    # Stride for moving to the next element in the row / column of tensor a
    a_col_stride = a_strides[1]
    a_row_stride = a_strides[
        2
    ]  # as mutiplication needs all the elements in the row for tensor a

    # Stride for moving to the next element in the row / column of tensor b
    b_col_stride = b_strides[
        1
    ]  # as mutiplication needs all the elements in the column for tensor b
    b_row_stride = b_strides[2]

    # The dimension for the result of each batch (must match: last dim of a, second-to-last of b)
    result_dim = b_shape[-2]

    # Process each batch of the output tensor in parallel
    for batch_index in prange(out_shape[0]):
        # Process each element in the output tensor for the current batch
        for row in range(out_shape[1]):
            for col in range(out_shape[2]):
                # Calculate the first element in the row of tensor a for the current batch
                a_index = batch_index * a_batch_stride + row * a_col_stride

                # Calculate the first element in the column of tensor b for the current batch
                b_index = batch_index * b_batch_stride + col * b_row_stride

                # Calculate the position of the result in the output tensor for the current batch, row and column
                out_index = (
                    batch_index * out_strides[0]
                    + row * out_strides[1]
                    + col * out_strides[2]
                )

                # Decalre a variable for the result of the products
                result = 0.0

                # Inner product loop for the calculating the sum of the products of different parts of elements in tensor a and b
                for _ in range(result_dim):
                    # Add the product of the elements pair in tensor a and b to the result
                    result += a_storage[a_index] * b_storage[b_index]

                    # Update the indices for the next element in the row of tensor a and the next element in the column of tensor b
                    a_index += a_row_stride
                    b_index += b_col_stride

                # Store the result in the output tensor storage
                out[out_index] = result


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
