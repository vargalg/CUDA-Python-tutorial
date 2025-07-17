from numba import cuda
import numpy as np

# Just some constants
N = 1024                    # Length of the vector to process
threads_per_block = 256     # Number of threads in each block

# Kernel to run on the GPU
@cuda.jit
def vector_addition(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]


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

    # Call kernel
    vector_addition[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result to host
    c = d_c.copy_to_host()

    # Print result
    print("Result:", c)
