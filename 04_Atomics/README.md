# Atomics

CUDA provides atomic memory operations:
- Basic operations that update memory safely.
- Threads are synchronized if they access the same location.
- But global atomics are slow:
	- Each thread waits while others read (~1000 cycles), update (fast), and write back (~1000 cycles).

## Dot Product with Atomic Operation (Global Accumulation)

We can simplify the previous program by using atomic operations.

### Modified Kernel
``` python
@cuda.jit
def dot_kernel(a, b, c):
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

    # Atomic accumulation into global memory
    if tid == 0:
        cuda.atomic.add(c, 0, smem[0])

    return
```

And host code:
```python
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
```

Here, `cuda.atomic.add(...)` ensures only one thread updates c[0] at a time.

This removes the race condition, but atomic operations serialize access, so performance may be slower than shared memory or CPU-based summation.

### Pros and Cons
| Property | Evaluation |
| -------- | :--------: |
| Correct result | ✅ |
| Simplicity | ✅ |
| Scalability | ✅ |
| Performance | ✅ but slower than shared-memory reduction |

🎓 Summary

| Approach | Correct? | Fast? | Complexity |
| -------- | :--------: | :--------: | :---------: |
| ❌ Naive global write | ❌ | ✅ | ❌ trivial |
| ✅ Partial sums, CPU reduces | ✅ | ⚠️ | ✅ almost |
| ✅ Shared memory + CPU reduce | ✅ | ✅ | ⚠️ more complex |
| ✅ Atomics global write | ✅ | ✅ | ✅ simpler |