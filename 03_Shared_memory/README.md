# Implementing More Complex Tasks on the GPU

Example programs are available here:  
https://github.com/vargalg/CUDA-Python-tutorial/tree/main/03_Shared_memory

---

## 1. Dot Product on CPU

The dot product of two vectors is the sum of the elementwise multiplications of two vectors of the same length:

$$ \text{dot}(a, b) = \sum_{i=0}^{N-1} a_i \cdot b_i $$

This is a sequential operation, easy to implement on CPU either with NumPy or manually with a `for` loop.

```python
import numpy as np

# Size of data to process
N = 1024*1024

# Function to calculate dot product.
def dot_cpu(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Entry point
if __name__ == '__main__':
    # memory allocations
    a = (np.ones(N) * 0.5).astype(np.float32)
    b = (np.ones(N) * 2.0).astype(np.float32)

    # Do calculation
    result = dot_cpu(a, b)

    # Result should be equal to N
    print(f"Dot product (CPU): {result}    (Should be: {N})")
```

### Explanation

Each iteration multiplies one pair of elements: a[i] * b[i].

Products are accumulated in the variable result.

Operations are sequential, one after the other.

Thanks to CPU caches and SIMD instructions, this can still be fast — but it does not scale well for very large datasets.

## Naive GPU Implementation

The following version is a naive CUDA kernel, which seems to perform the same calculation on GPU.

```python
import numpy as np
from numba import cuda

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
```

### Why is it wrong?

- c[0] is the same global memory address for all threads.
- Thousands of threads try to modify it simultaneously.
- There is no synchronization (no atomic operation), so writes overlap → race condition.
- The result is non-deterministic and may differ on each run (officially "undefined" in CUDA).

### Lessons learned

- Not all operations are easily parallelizable.
- Memory access and synchronization are key issues in CUDA programming.

## GPU Partial Sums in Global Memory

One simple fix: let each thread write its own partial result into a separate array, and let the CPU sum them at the end.

### Idea

- Each thread computes its own a[i] * b[i].
- The result is written into a separate array (partial_sums).
- The CPU computes the final result with sum().
- A thread can process more than one element, using a stride.

This avoids race conditions, since no two threads write to the same memory address.

### CUDA Kernel (working but not optimized)
```python
import numpy as np
from numba import cuda

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

    # Save the partial sum we calculated
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
```

### Explanation
Each thread only writes to its own memory slot → no conflicts.

Correct result, but performance is not optimal:

- Many global memory writes.
- The GPU does not combine results — CPU does.
- Works fine, but inefficient for large datasets.

## Optimized Dot Product with Shared Memory

CUDA multiprocessors have a special memory region called shared memory:
- Shared by all threads of a block.
- Limited in size (depends on Compute Capability).
- Very fast (~a few clock cycles).
We can use it to accelerate reduction.

Each block computes its own partial sum using shared memory, then writes that to global memory.
Finally, the CPU sums the block results.

### Idea

1. Each thread computes its own a[i] * b[i].
2. Stores it in a **shared memory array**.
3. The block reduces its values using parallel reduction.
4. Thread 0 writes the block’s result into global memory.
5. CPU sums across all blocks.

### CUDA Kernel
```python
@cuda.jit
def dot_kernel(a, b, c_partial):
    # Shared memory for partial sums inside the block
    smem = cuda.shared.array(threads_per_block, numba.float32)

    # Get index and grid size
    index = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Calculate sum of every M-th element
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

    # Save the partial sum computed by this block
    if tid == 0:
        c_partial[cuda.blockIdx.x] = smem[0]

    return
```

And the host code:

```python
import numpy as np
import numba
from numba import cuda

N = a.size
threads_per_block = 256

# Entry point
if __name__ == '__main__':
    # memory allocations
    a = (np.ones(N) * 0.5).astype(np.float32)
    b = (np.ones(N) * 2.0).astype(np.float32)
    c_partial = (np.zeros(blocks_per_grid)).astype(np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c_partial = cuda.to_device(c_partial)

    # Call kernel
    dot_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c_partial)

    # Copy result to host
    d_c_partial.copy_to_host(ary=c_partial)

    c = np.sum(c_partial)

    # Result should be equal to N
    print(f"Dot product (GPU implementation: PartSum + Shared Mem.): {c}    (Should be: {N})")
```

### Explanation
Shared memory is very fast since all threads in a block run on the same multiprocessor.

`cuda.syncthreads()` ensures no thread starts reduction before all have stored their value.

The reduction loop is logarithmic: O(log n).

### Advantages

- No race conditions.
- Less global memory traffic.
- Faster than previous versions.
- Classic CUDA “pattern” (parallel reduction), useful in many algorithms.
