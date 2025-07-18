from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

# Kép mérete
height = 512
width = 512
frequency = 5.0

# Blokk méret
threadsperblock = (16, 16)


@cuda.jit
def generate_sine_wave(output, frequency):
    x, y = cuda.grid(2)

    height = output.shape[0]
    width = output.shape[1]

    if x < width and y < height:
        output[y, x] = np.sin(2 * np.pi * frequency * x / width) * np.sin(2 * np.pi * frequency * y / height)

if __name__ == '__main__':
    # Kimeneti tömb
    output = np.zeros((height, width), dtype=np.float32)
    d_output = cuda.to_device(output)

    # Rácsméret kiszámítása
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Kernel hívása
    generate_sine_wave[blockspergrid, threadsperblock](d_output, frequency)

    # Eredmény visszamásolása
    output = d_output.copy_to_host()

    # Kirajzolás
    plt.imshow(output, cmap='gray')
    plt.title("2D szinuszhullám")
    plt.colorbar()
    plt.show()
