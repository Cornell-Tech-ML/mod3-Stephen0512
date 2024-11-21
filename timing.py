import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np

# Initialize tensor backends
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)  # CPU backend
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)  # GPU backend


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Run a single matrix multiplication test using the specified backend.

    Args:
    ----
        backend: The tensor backend to use (FastTensorBackend or GPUBackend)
        size: Size of the square matrices to multiply (default: 16)

    Returns:
    -------
        None

    """
    batch_size = 2

    # Create random input tensors
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y  # Perform matrix multiplication


if __name__ == "__main__":
    # Warmup runs to initialize backends
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    # Number of trials to run for each size
    ntrials = 3
    times = {}

    # Test different matrix sizes
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []

        # Run multiple trials for each size
        for _ in range(ntrials):
            # Time Fast backend
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            # Time GPU backend
            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            # Calculate execution times
            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            # Store times for averaging
            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        # Calculate and store average times
        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    # Print timing summary
    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

