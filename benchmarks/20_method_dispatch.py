# Method Dispatch Benchmark - Tests class method calls
# Stresses method lookup and function calls

import time


class Calculator:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def compute(x):
        return Calculator.add(Calculator.mul(x, 2), Calculator.sub(x, 1))


def method_benchmark(n):
    total = 0
    for i in range(n):
        total += Calculator.compute(i)
    return total


start = time.time()
result = method_benchmark(100000)
end = time.time()

print(f"Method dispatch (100000 iterations): {result}")
print(f"Time: {end - start} seconds")
