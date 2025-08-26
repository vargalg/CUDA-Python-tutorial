# Pytorch Integration

In this lesson, we show how to embed a custom CUDA kernel into a PyTorch model so that the new operation can be used as a PyTorch `nn.Module`.  
The example uses **Numba**, which can work directly with PyTorch `cuda.Tensor`s.

Example code is available here:  
https://github.com/vargalg/CUDA-Python-tutorial/blob/main/05_Pytorch_interoperability/0501_torch_demo.py

---

## Goal

Create a PyTorch layer that adds `+2` to every input element using a **custom CUDA kernel**.

---

## 1. CUDA Kernel with Numba

A CUDA kernel can be defined with the `@cuda.jit` decorator:

```python
@cuda.jit
def add_two_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] += 2
```

This kernel adds 2 to every element.

## 2. Creating a Custom Function

PyTorch operations can be implemented by subclassing torch.autograd.Function, which defines the required forward (activation) and backward (gradient) operations.

```python
class AddTwoNumbaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        assert input_tensor.is_cuda, "Input tensor must be on CUDA"
        assert input_tensor.dtype == torch.float32, "Tensor must be float32"
        assert input_tensor.is_contiguous(), "Tensor must be contiguous"

        device_array = cuda.as_cuda_array(input_tensor)

        threads_per_block = 128
        blocks_per_grid = (device_array.size + threads_per_block - 1) // threads_per_block

        add_two_kernel[blocks_per_grid, threads_per_block](device_array)
        cuda.synchronize()

        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Currently we do not modify the gradient – pass it through unchanged
        return grad_output
```

**Important notes:**

- `input_tensor.is_contiguous()` → Numba can only handle contiguous memory.
- `cuda.as_cuda_array(tensor)` → converts a PyTorch cuda.Tensor into a Numba-compatible device array (zero-copy).
- The `backward()` method here simply passes through the gradient. Since the operation is just a constant shift, it is not differentiable, and we are not modifying the gradient.

## 3. nn.Module Wrapper
To fit PyTorch’s layer structure, we write an nn.Module wrapper:

```python
class AddTwoLayer(nn.Module):
    def forward(self, x):
        return AddTwoNumbaFunction.apply(x)
```

## 4. Usage Example

```python
if __name__ == '__main__':
    x = torch.ones(5, device='cuda', dtype=torch.float32)
    layer = AddTwoLayer()

    y = layer(x)
    print(y)  # [3.0, 3.0, 3.0, 3.0, 3.0]
```