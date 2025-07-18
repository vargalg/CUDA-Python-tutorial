import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import timeit


# Kép paraméterek
width, height = 1024, 1024
zoom = 1.0
move_x, move_y = 0.0, 0.0
c_re, c_im = -0.7, 0.27015  # Julia-halmaz konstans
max_iter = 300

# CUDA Blokk méretek
threadsperblock = (16, 16)

# CUDA események létrehozása
start = cuda.event()
end = cuda.event()


@cuda.jit
def julia_kernel(output, width, height, zoom, move_x, move_y, c_re, c_im, max_iter):
    x, y = cuda.grid(2)

    if x >= width or y >= height:
        return

    # Koordináták leképezése a komplex síkra
    zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + move_x
    zy = (y - height / 2) / (0.5 * zoom * height) + move_y

    iteration = 0
    while zx*zx + zy*zy < 4.0 and iteration < max_iter:
        xtemp = zx*zx - zy*zy + c_re
        zy = 2.0*zx*zy + c_im
        zx = xtemp
        iteration += 1

    output[y, x] = iteration


def generate_julia_gpu():
    # Kimeneti mátrix
    output = np.zeros((height, width), dtype=np.uint16)
    d_output = cuda.to_device(output)

    # CUDA paraméterek
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Kernel indítása
    start.record()
    julia_kernel[blockspergrid, threadsperblock](d_output, width, height, zoom, move_x, move_y, c_re, c_im, max_iter)
    end.record()

    # Eredmény visszamásolása
    output = d_output.copy_to_host()

    return output

def generate_julia_cpu():
    output = np.zeros((height, width), dtype=np.uint16)

    for y in range(height):
        zy = (y - height / 2) / (0.5 * zoom * height) + move_y
        for x in range(width):
            zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + move_x

            iteration = 0
            while zx * zx + zy * zy < 4.0 and iteration < max_iter:
                xtemp = zx * zx - zy * zy + c_re
                zy = 2.0 * zx * zy + c_im
                zx = xtemp
                iteration += 1

            output[y, x] = iteration

    return output

if __name__ == '__main__':

    gpu_first_time = timeit.timeit(generate_julia_gpu, number=1)
    gpu_second_time = timeit.timeit(generate_julia_gpu, number=1)
    output_GPU = generate_julia_gpu()

    gpu_kernel_only_time = cuda.event_elapsed_time(start, end)/1000.0

    cpu_time = timeit.timeit(generate_julia_cpu, number=1)
    output_CPU = generate_julia_gpu()

    print(f"First GPU time: {gpu_first_time}")
    print(f"Second GPU time: {gpu_second_time}")
    print(f"Second GPU kernel time: {gpu_kernel_only_time}")
    print(f"CPU time: {cpu_time}")

    # Kirajzolás
    plt.figure()
    plt.imshow(output_GPU, cmap='inferno', extent=[0, width, 0, height])
    plt.title(f"Julia Set on GPU (c = {c_re} + {c_im}i)")
    plt.axis('off')

    plt.figure()
    plt.imshow(output_CPU, cmap='inferno', extent=[0, width, 0, height])
    plt.title(f"Julia Set on CPU (c = {c_re} + {c_im}i)")
    plt.axis('off')

    plt.show()
