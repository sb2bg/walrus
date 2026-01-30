# Fibonacci Iterative Benchmark - Tests loop performance
# Calculates many fibonacci numbers iteratively

import time


def fib_iter(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b


start = time.time()
total = 0
for i in range(100000):
    total += fib_iter(50)
end = time.time()

print(f"100000 iterations of fib_iter(50)")
print(f"Total sum: {total}")
print(f"Time: {end - start} seconds")
