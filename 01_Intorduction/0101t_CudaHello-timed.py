from traceback import print_tb

from numba import cuda
import numpy as np
import timeit


# Just some constants
N = 1024                    # Length of the vector to process
threads_per_block = 256     # Number of threads in each block

# Kernel to run on the GPU
@cuda.jit
def vector_addition(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]


def exec_kernel_only(a, b, c, grid_size, block_size):
    def _exec_kernel_only():
        vector_addition[grid_size, block_size](a, b, c)
        cuda.synchronize()

    return _exec_kernel_only


def exec_with_copy(a, b, c, d_a, d_b, d_c,  grid_size, block_size):
    def _exec_with_copy():
        cuda.to_device(a, to=d_a)
        cuda.to_device(b, to=d_b)

        vector_addition[grid_size, block_size](d_a, d_b, d_c)

        d_c.copy_to_host(ary=c)

    return _exec_with_copy

def exec_with_create_and_copy(a, b, grid_size, block_size):
    def _exec_with_create_and_copy():
        # Create GPU-counterparts of variables on the device
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array(N, dtype=np.float32)

        vector_addition[grid_size, block_size](d_a, d_b, d_c)

        # Copy result to host
        c = d_c.copy_to_host()

        return c
    return _exec_with_create_and_copy

if __name__ == '__main__':
    # Create variables on the host
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Create GPU-counterparts of variables on the device
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array(N, dtype=np.float32)

    # Calculate how many blocks are needed to cover all the data elements
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Run kernel once to compile it
    timing_result = timeit.Timer(exec_kernel_only(d_a, d_b, d_c, blocks_per_grid, threads_per_block)).timeit(1)
    print(f"Time of first kernel execution: {timing_result}")

    timing_result = timeit.Timer(exec_kernel_only(d_a, d_b, d_c, blocks_per_grid, threads_per_block)).timeit(10)
    print(f"Time of kernel execution only: {timing_result}")

    timing_result = timeit.Timer(exec_with_copy(a, b, c, d_a, d_b, d_c, blocks_per_grid, threads_per_block)).timeit(10)
    print(f"Time of kernel execution only: {timing_result}")

    timing_result = timeit.Timer(exec_with_create_and_copy(a, b, blocks_per_grid, threads_per_block)).timeit(10)
    print(f"Time of kernel execution only: {timing_result}")

    # Call kernel
    vector_addition[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result to host
    c = d_c.copy_to_host()

    # Print result
    print("Result:", c)
