# Fibonacci Recursive Benchmark - Tests function call overhead and recursion
# This is deliberately naive (exponential) to stress test call stack

import time

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

start = time.time()
result = fib(30)
end = time.time()

print(f"Fibonacci(30) = {result}")
print(f"Time: {end - start} seconds")
