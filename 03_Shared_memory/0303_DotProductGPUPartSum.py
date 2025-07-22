import numpy as np
from numba import cuda
import timeit


# Just some constants
N = 1024*1024
threads_per_block = 256     # Number of threads in each block
blocks_per_grid = 32


# Kernel function to calculate dot product.
@cuda.jit
def dot_kernel(a, b, c_partial):
    # Get index and grid size
    index = cuda.grid(1)
    stride = cuda.gridsize(1)

    # calculate sum of every M-th element
    part_sum = 0.0
    while index < a.shape[0]:
        part_sum += a[index] * b[index]
        index += stride

    # Save the part of the sum that we calculated
    c_partial[cuda.grid(1)] = part_sum

    return


# Entry point
if __name__ == '__main__':
    # Calculate total number of threads
    num_total_threads = blocks_per_grid * threads_per_block

    # memory allocations
    a = (np.ones(N) * 0.5).astype(np.float32)
    b = (np.ones(N) * 2.0).astype(np.float32)
    c_partial = (np.zeros(num_total_threads)).astype(np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c_partial = cuda.to_device(c_partial)

    # Call kernel
    dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c_partial)

    # Copy result to host
    d_c_partial.copy_to_host(ary=c_partial)

    c = np.sum(c_partial)

    # Result should be equal to N
    print(f"Dot product (GPU implementation: PartSum): {c}    (Should be: {N})")


    def single_run():
        dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c_partial)

    execution_time = timeit.timeit(single_run, number=10)

    print("Execution time:", execution_time)
