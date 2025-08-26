# CUDA Python: Introduction

In these tutorials, we will use the **Numba** library to write GPU code for the CUDA platform.

## The Numba Library

### What is Numba?
Numba is a JIT (Just-In-Time) compiler that allows us to translate Python code into machine code at runtime.  
It optimizes Python computations—especially NumPy-based array operations—and can also generate code that runs on CUDA GPUs, provided an NVIDIA GPU and the CUDA toolchain are available.

### Why Numba?
- Python stays Python, but uses CUDA: no need to learn C++ or CUDA C.  
- Fast prototyping for the GPU.  
- Excellent integration with NumPy and the Python ecosystem.  
- Easy to integrate into existing image processing projects.  

### Installation

```bash
pip install numba
```

More details: [Numba installation guide](https://numba.readthedocs.io/en/stable/user/installing.html)

### Requirements for CUDA support

- An NVIDIA GPU (Compute Capability ≥ 3.0)
- Installed CUDA Toolkit (e.g., nvcc --version should work)
- A compatible NVIDIA driver (required for using the numba.cuda module)