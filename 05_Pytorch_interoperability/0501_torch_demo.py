import torch
import torch.nn as nn
from numba import cuda

@cuda.jit
def add_two_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] += 2


class AddTwoNumbaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        assert input_tensor.is_cuda, "Input tensor must be on CUDA"
        assert input_tensor.dtype == torch.float32, "Tensor must be float32"
        assert input_tensor.is_contiguous(), "Tensor must be contiguous"

        # Create a Numba-compatible view of the PyTorch tensor
        device_array = cuda.as_cuda_array(input_tensor)

        # Determine grid and block dimensions
        threads_per_block = 128
        blocks_per_grid = (device_array.size + threads_per_block - 1) // threads_per_block

        # Launch the kernel
        add_two_kernel[blocks_per_grid, threads_per_block](device_array)

        # Synchronize to ensure completion
        cuda.synchronize()

        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Pass through the gradient unmodified
        return grad_output

class AddTwoLayer(nn.Module):
    def forward(self, x):
        return AddTwoNumbaFunction.apply(x)


if __name__ == '__main__':
    x = torch.ones(5, device='cuda', dtype=torch.float32)
    layer = AddTwoLayer()
    model = nn.Sequential(layer, layer)

    y = model(x)
    print(y)  # Should be all 3's