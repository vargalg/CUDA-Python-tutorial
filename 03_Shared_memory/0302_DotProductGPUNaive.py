"""
    WARNING: Code might result in wrong result!
"""

import numpy as np
from numba import cuda
import timeit

# Just some constants
N = 1024*1024
threads_per_block = 256     # Number of threads in each block


# Kernel function to calculate dot product.
@cuda.jit
def dot_kernel(a, b, c):
    index = cuda.grid(1)

    if index < a.shape[0]:
        c[0] += a[index] * b[index]

    return


# Entry point
if __name__ == '__main__':
    # memory allocations
    a = (np.ones(N) * 0.5).astype(np.float32)
    b = (np.ones(N) * 2.0).astype(np.float32)
    c = np.zeros(1, dtype=np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    # Calculate how many blocks are needed to cover all the data elements
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Call kernel
    dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result to host
    d_c.copy_to_host(ary=c)

    # Result should be equal to N
    print(f"Dot product (Naive GPU implementation): {c}    (Should be: {N})")


    def single_run():
        dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    execution_time = timeit.timeit(single_run, number=10)

    print("Execution time:", execution_time)
