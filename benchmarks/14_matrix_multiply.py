# Matrix Multiplication Benchmark - Tests nested loops and array access
# Classic O(n³) algorithm

import time

def create_matrix(size, seed):
    matrix = []
    current = seed
    for i in range(size):
        row = []
        for j in range(size):
            current = (current * 1103515245 + 12345) % 2147483648
            row.append(current % 100)
        matrix.append(row)
    return matrix

def matrix_multiply(a, b, size):
    result = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            sum_val = 0
            for k in range(size):
                sum_val += a[i][k] * b[k][j]
            result[i][j] = sum_val
    return result

size = 50
start = time.time()
a = create_matrix(size, 42)
b = create_matrix(size, 123)
c = matrix_multiply(a, b, size)
end = time.time()

print(f"Matrix multiply {size}x{size}")
print(f"Result[0][0] = {c[0][0]}")
print(f"Time: {end - start} seconds")
