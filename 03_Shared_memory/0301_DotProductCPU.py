import numpy as np
import timeit

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

    def single_run():
        dot_cpu(a, b)

    execution_time = timeit.timeit(single_run, number=1)

    print("Execution time:", execution_time)
