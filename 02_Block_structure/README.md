# Cuda Block Structure

Example source code can be found here:
[https://github.com/vargalg/CUDA-Python-tutorial/tree/main/02_Block_structure](https://github.com/vargalg/CUDA-Python-tutorial/tree/main/02_Block_structure)

## What is a Block Structure?

The CUDA architecture organizes parallel execution units hierarchically:

- The program is executed by **threads**.  
- Threads are grouped into **blocks**.  
- Blocks form **grids**.  

![c0201grid.png](https://wiki.varex.hu/assets/cuda/en0201blockstructure.png)

CUDA programs are written so that each thread processes a portion of the input data based on its own index.  
This makes it possible to process large datasets efficiently on the GPU.

## CUDA Thread Indexing

Each CUDA thread has three types of local identifiers:

| Variable             | Meaning |
|:---------------------|:--------|
| `cuda.threadIdx.x`   | Index of the current thread within its block |
| `cuda.blockIdx.x`    | Index of the current block within the grid |
| `cuda.blockDim.x`    | Number of threads in a block |
| `cuda.gridDim.x`     | Number of blocks in a grid |

The **global thread index** can be computed as:

```python
global_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

## 1D Block Structure – Vector Addition

**Goal:** element-wise addition of two vectors of the same size.

The function `cuda.grid(1)` returns the global index of the current thread, independent of its block.

```python
from numba import cuda
import numpy as np

# Just some constants
N = 1024                    # Length of the vector to process
threads_per_block = 128     # Number of threads in each block

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
```

### Explanation

- `threads_per_block = 128`: each block contains 128 threads.
- `blocks_per_grid = ...`: computes how many blocks are needed to cover all elements.
- `cuda.grid(1)`: computes the global thread index.

Each thread only executes if idx < a.size, to avoid out-of-range access.

### Exercise

- Try changing the number of threads per block (from 1 up to the maximum your GPU allows) and measure how the runtime changes!

For this exercise, you can use the example in the repository:
`02_Block_structure/0201t_CudaHello-timed.py`

## Why Do We Need This?

Because that’s simply how the GPU is built:

- Inside the GPU, ALUs are grouped into Streaming Multiprocessors (SMs).
- A GPU consists of multiple multiprocessors.

The program’s structure follows this hardware: ALU = Thread, Multiprocessor = Block, GPU = Grid.

![c0202gridvsgpu.png](https://wiki.varex.hu/assets/cuda/en0202gridandgpu.png)


### Note

Usually the grid is much larger than the GPU. The GPU handles this by:
- During execution, some blocks are loaded into the multiprocessors.
- A block starts, runs, finishes, and then gives its place to another block. (It never restarts.)
- Within a multiprocessor, only a subset of the threads run simultaneously (typically 32 active threads = a warp).
- Thread and block scheduling is managed by the GPU hardware and driver. We have little (if any) control over it.


## 2D Block Structure (Introduction)
A 2D grid structure is useful for processing two-dimensional data (e.g., images, matrices).

CUDA also supports 2D and 3D grids and blocks:

```python
x, y = cuda.grid(2)
```

This gives the (x, y) coordinates of the current thread in the grid.

Theory: 2D Global Thread Index
The 2D block and grid structure can be described with these variables:

| Variable | Meaning |
|:--------|:--------|
| `cuda.threadIdx.x/y/z`	| Thread position within the block (x/y/z) |
| `cuda.blockIdx.x/y/z`	| Block position within the grid (x/y/z) |
| `cuda.blockDim.x/y/z`	| Number of threads in a block (x/y/z) |
| `cuda.gridDim.x/y/z`	| Number of blocks in the grid (x/y/z) |

Global indices:

```python
x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
```

Alternative (recommended):

```python
x, y = cuda.grid(2)
```


### Example: 2D Sine Wave

```python
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def generate_sine_wave(output, frequency):
    x, y = cuda.grid(2)

    height = output.shape[0]
    width = output.shape[1]

    if x < width and y < height:
        output[y, x] = np.sin(2 * np.pi * frequency * x / width) * np.sin(2 * np.pi * frequency * y / height)

# Image size
height = 512
width = 512
frequency = 5.0

# Output array
output = np.zeros((height, width), dtype=np.float32)
d_output = cuda.to_device(output)

# Block size and grid size
threadsperblock = (16, 16)
blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Kernel launch
generate_sine_wave[blockspergrid, threadsperblock](d_output, frequency)

# Copy result back to host
output = d_output.copy_to_host()

# Visualization
plt.imshow(output, cmap='gray')
plt.title("2D Sine Wave")
plt.colorbar()
plt.show()
```

#### What did we see?
- `cuda.grid(2)` automatically computes (x, y) coordinates.
- `threadsperblock` and blockspergrid are now tuples, not scalars.
- Each thread computes a single pixel’s value.
- The result is a 2D sinusoidal pattern, which can be visually checked.

### Interesting Examples

- Here https://github.com/vargalg/CUDA-Python-tutorial/blob/main/02_Block_structure/0203_Julia_set.py
- And here: https://github.com/vargalg/CUDA-Python-tutorial/blob/main/02_Block_structure/0204_SecretIdentity.py

## One More Note (32-bit Floats)

Throughout the examples, data is consistently created as float32.
Let’s see what happens if we omit type specifications or use float64 instead!

### Explanation

GPUs have a very large number of ALUs optimized for float32. But there are far fewer units capable of processing 64-bit floats (or integers).

Therefore, GPUs are most efficient with 32-bit floating-point operations. Any other data type will be significantly slower—unless you are using a high-end server GPU with many double-precision units.