import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
from math import sqrt

# Image size, and coordinate ranges
width, height = 1024, 768
x_min, x_max = -7.0, 7.0
y_min, y_max = -5.0, 5.0

# Size of CUDA blocks
threadsperblock = (16, 16)


@cuda.jit(device=True)
def equation(x, y):
    x2 = x * x
    y2 = y * y
    abs_x = abs(x)

    # Just some equation
    tf = ( x * x / 49 + y * y / 9 - 1 < 0 and
           ((abs(x) >= 4 and (-3 * sqrt(33.0) / 7 <= y and y <= 0)) or
           (abs(x) >= 3 and y >= 0)) or
         ((-3 <= y and y <= 0) and
          (-4 <= x and x < 4) and
          (abs(x / 2.0) + sqrt(1.0 - (abs(abs(x) - 2) - 1) * (abs(abs(x) - 2) - 1)) - (3 * sqrt(33.0) - 7) * x * x / 112.0 - y - 3.0) <= 0) or
           (y >= 0 and
            (3.0 / 4.0 <= abs(x) and abs(x) <= 1) and
            -8 * abs(x) - y + 9 >= 0) or
           ((1.0 / 2.0 <= abs(x) and abs(x) <= 3.0 / 4.0) and
            3.0 * abs(x) - y + 3.0 / 4.0 >= 0 and
            y >= 0) or
           (abs(x) <= 1.0 / 2.0 and
            y > 0 and
            9.0 / 4.0 - y >= 0) or
           (abs(x) >= 1 and
            y >= 0 and
            (-abs(x) / 2.0-3.0 / 7.0 * sqrt(10.0) * sqrt(4.0 - (abs(x) - 1.0) * (abs(x) - 1.0)) - y + 6.0 * sqrt(
                       10.0) / 7.0 + 3.0 / 2.0) >= 0)
    );

    if tf:
        return 1

    return 0


@cuda.jit
def render(output, x_min, x_max, y_min, y_max):
    i, j = cuda.grid(2)
    height, width = output.shape

    if i >= height or j >= width:
        return

    # Koordináták leképezése a komplex síkra
    x = x_min + (x_max - x_min) * j / width
    y = y_max - (y_max - y_min) * i / height

    if equation(x, y):
        output[i, j] = 255
    else:
        output[i, j] = 0

if __name__ == '__main__':
    # Allocate memory array for output
    output = np.zeros((height, width), dtype=np.uint8)
    d_output = cuda.to_device(output)

    blockspergrid_x = (width + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_y = (height + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    render[blockspergrid, threadsperblock](d_output, x_min, x_max, y_min, y_max)
    output = d_output.copy_to_host()

    # Megjelenítés
    plt.imshow(output, cmap='gray', extent=[x_min, x_max, y_min, y_max])
    plt.title("Call the saviour (CUDA)")
    plt.axis('off')
    plt.show()
