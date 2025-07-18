import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


# Kép paraméterek
width, height = 1024, 1024
zoom = 1.0
move_x, move_y = 0.0, 0.0
c_re, c_im = -0.7, 0.27015  # Julia-halmaz konstans
max_iter = 300

# CUDA Blokk méretek
threadsperblock = (16, 16)


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


if __name__ == '__main__':
    # Kimeneti mátrix
    output = np.zeros((height, width), dtype=np.uint16)
    d_output = cuda.to_device(output)

    # CUDA paraméterek
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Kernel indítása
    julia_kernel[blockspergrid, threadsperblock](d_output, width, height, zoom, move_x, move_y, c_re, c_im, max_iter)

    # Eredmény visszamásolása
    output = d_output.copy_to_host()

    # Kirajzolás
    plt.imshow(output, cmap='inferno', extent=[0, width, 0, height])
    plt.title(f"Julia Set (c = {c_re} + {c_im}i)")
    plt.axis('off')
    plt.show()
