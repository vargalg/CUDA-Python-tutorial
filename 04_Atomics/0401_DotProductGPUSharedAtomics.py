import numpy as np
import numba
from numba import cuda
import timeit


# Just some constants
N = 1024*1024
threads_per_block = 256     # Number of threads in each block
blocks_per_grid = 32


# Kernel function to calculate dot product.
@cuda.jit
def dot_kernel(a, b, c):
    # shared memory a blokk részösszegének gyűjtéséhez
    smem = cuda.shared.array(threads_per_block, numba.float32)

    # Get index and grid size
    index = cuda.grid(1)
    stride = cuda.gridsize(1)

    # calculate sum of every M-th element
    part_sum = 0.0
    while index < a.shape[0]:
        part_sum += a[index] * b[index]
        index += stride

    smem[cuda.threadIdx.x] = part_sum

    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    tid = cuda.threadIdx.x

    while s > 0:
        if tid < s:
            smem[tid] += smem[tid+s]
        cuda.syncthreads()
        s //= 2

    # Save the part of the sum that the threads of the block have calculated
    if tid == 0:
        cuda.atomic.add(c, 0, smem[0])

    return


# Entry point
if __name__ == '__main__':
    # memory allocations
    a = (np.ones(N) * 0.5).astype(np.float32)
    b = (np.ones(N) * 2.0).astype(np.float32)
    c = (np.zeros(1)).astype(np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    # Call kernel
    dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result to host
    d_c.copy_to_host(ary=c)

    # Result should be equal to N
    print(f"Dot product (GPU implementation: PartSum + Shared Mem. + Atomics): {c}    (Should be: {N})")

    def single_run():
        dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    execution_time = timeit.timeit(single_run, number=10)

    print("Execution time:", execution_time)
