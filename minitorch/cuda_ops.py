# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator that compiles a function for execution on CUDA device."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator that compiles a function as a CUDA kernel."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """CUDA implementation of element-wise binary operations between tensors.

        Args:
        ----
            fn: Binary function that takes two floats and returns a float.
                This function will be applied element-wise to the input tensors.

        Returns:
        -------
            A function that takes two tensors and returns a new tensor containing
            the element-wise application of fn. The output shape is the broadcast
            shape of the input tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """CUDA implementation of tensor reduction along a dimension.

        Args:
        ----
            fn: Binary reduction function that takes two floats and returns a float.
                This function should be associative and commutative for correctness.
            start: Initial value for the reduction (default: 0.0)

        Returns:
        -------
            A function that takes a tensor and a dimension index, and returns a new tensor
            with that dimension reduced using the specified function.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """CUDA implementation of batched matrix multiplication.

        This method performs matrix multiplication between two tensors using CUDA.

        Args:
        ----
            a: First input tensor with shape (..., n, m)
            b: Second input tensor with shape (..., m, p)
               The batch dimensions (...) must be broadcastable

        Returns:
        -------
            Output tensor with shape (..., n, p) where ... is the broadcasted
            batch dimensions

        Raises:
        ------
            AssertionError: If the inner dimensions don't match (a.shape[-1] != b.shape[-2])

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Each CUDA thread handles one element
        if i < out_size:
            # Calculate output index for this thread
            to_index(i, out_shape, out_index)

            # Calculate input index accounting for broadcasting using the output index calculated above
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get positions in tensor storages for the input and output
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)

            # Apply function and store result in parallel
            out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Each CUDA thread handles one element
        if i < out_size:
            # Calculate output index for this thread
            to_index(i, out_shape, out_index)

            # Calculate input indices accounting for broadcasting using the output index calculated above
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Get positions in tensor storages for the two inputs and output
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)

            # Apply function to inputs and store result in parallel
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Initialize shared memory with input values or zero if out of bounds
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    # Synchronize threads to ensure shared memory is fully populated
    cuda.syncthreads()

    # Iteratively sum pairs of elements, reducing array size by half each time
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and i + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Store final sum for this block in output array
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum kernel to prepare for reduce.

    This function takes a tensor and applies the sum reduction kernel to it using CUDA.
    It divides the input tensor into blocks and sums the elements within each block
    in parallel using shared memory.

    Args:
    ----
        a (Tensor): Input tensor to be reduced

    Returns:
    -------
        TensorData: Output tensor containing partial sums from each block

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Each block processes one output position. Exit if this block's position exceeds output size
        if out_pos >= out_size:
            return

        # Initialize each thread's shared memory slot with the starting reduction value
        cache[pos] = reduce_value

        # Get size of reduction dimension
        reduce_size = a_shape[reduce_dim]

        # Convert the flat output position to a multi-dimensional output index
        to_index(out_pos, out_shape, out_index)

        # Each thread in block handles different elements along reduction dimension
        # Thread 0 handles element 0, Thread 1 handles element 1, etc.
        out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

        # Convert the multi-dimensional output index to a flat position in the input storage
        start = index_to_position(out_index, a_strides)

        # Only process if this thread's element exists in input tensor
        if out_index[reduce_dim] < reduce_size:
            # Thread 0 combines new value with existing cache
            # Other threads just load their values directly
            cache[pos] = (
                fn(cache[pos], a_storage[start]) if pos == 0 else a_storage[start]
            )
            cuda.syncthreads()

            # Parallel reduction within shared memory
            # Each iteration combines pairs of elements, halving active threads
            stride = BLOCK_DIM // 2
            while stride > 0:
                if pos < stride:
                    # Active threads combine their value with one stride distance away
                    cache[pos] = fn(cache[pos], cache[pos + stride])
                # Ensure all threads complete before next iteration to avoid race conditions
                cuda.syncthreads()
                stride //= 2

        # Thread 0 writes the final reduced result for this block to global memory
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # Each thread handles one element of the output matrix
    # threadIdx.x determines the row, threadIdx.y determines the column
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Create shared memory arrays to store the input matrices A and B
    # This allows faster access compared to global memory
    shared_mem_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_mem_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Only threads corresponding to valid matrix elements should load data
    if row < size and col < size:
        # Map 2D coordinates (row,col) to 1D array index
        # Using row-major layout: index = row * width + col
        input_idx = row * size + col

        # Load corresponding elements from input matrices into shared memory
        shared_mem_a[row, col] = a[input_idx]
        shared_mem_b[row, col] = b[input_idx]

    # Synchronize to ensure all threads have finished loading shared memory
    cuda.syncthreads()

    # Only compute output for threads within matrix dimensions
    if row < size and col < size:
        # Initialize accumulator for dot product calculation
        result = 0.0

        # Compute dot product between row of A and column of B
        # by iterating through the k dimension shared by both matrices
        for k in range(size):
            result += shared_mem_a[row, k] * shared_mem_b[k, col]

        # Write the computed result back to global memory
        # Using same row-major indexing as input
        output_idx = row * size + col
        out[output_idx] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Wrapper function for matrix multiplication practice kernel.

    This function takes two input tensors and performs matrix multiplication using CUDA.
    It sets up the CUDA grid and block dimensions, allocates output memory, and calls
    the CUDA kernel to perform the actual computation.

    Args:
    ----
        a (Tensor): First input tensor of shape [size, size]
        b (Tensor): Second input tensor of shape [size, size]

    Returns:
    -------
        TensorData: Output tensor containing the matrix multiplication result
                    with shape [size, size]

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```

    Returns
    -------
        None: Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize accumulator for the dot product result
    accumulator = 0.0

    # Process the matrices in BLOCK_DIM x BLOCK_DIM tiles
    for block_pos in range(0, a_shape[-1], BLOCK_DIM):
        # Initialize shared memory tiles to zero
        a_shared[pi, pj] = 0
        b_shared[pi, pj] = 0

        # Synchronize to ensure all threads have finished loading shared memory
        cuda.syncthreads()

        # Load tile from matrix A into shared memory if within bounds
        # Each thread loads one element using its local position (pi,pj)
        if i < a_shape[-2] and (block_pos + pj) < a_shape[-1]:
            a_idx = (
                batch * a_batch_stride  # Batch offset
                + i * a_strides[-2]  # Row offset
                + (block_pos + pj) * a_strides[-1]  # Column offset
            )
            a_shared[pi, pj] = a_storage[a_idx]

        # Load tile from matrix B into shared memory if within bounds
        # Each thread loads one element using its local position (pi,pj)
        if (block_pos + pi) < b_shape[-2] and j < b_shape[-1]:
            b_idx = (
                batch * b_batch_stride  # Batch offset
                + (block_pos + pi) * b_strides[-2]  # Row offset
                + j * b_strides[-1]  # Column offset
            )
            b_shared[pi, pj] = b_storage[b_idx]

        # Synchronize to ensure all threads have finished loading shared memory
        cuda.syncthreads()

        # Compute partial dot product for this tile if output position is valid
        if i < out_shape[-2] and j < out_shape[-1]:
            # Only accumulate up to edge of current tile or matrix
            for k in range(min(BLOCK_DIM, a_shape[-1] - block_pos)):
                accumulator += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize to ensure all threads have finished loading shared memory
        cuda.syncthreads()

    # Write final accumulated result to global memory if position is valid
    if i < out_shape[-2] and j < out_shape[-1]:
        out_idx = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_idx] = accumulator


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
