# Function Call Stress Test - Tests function call overhead
# Tests many function calls (Walrus doesn't fully support closures)

import time

def make_adder(x):
    # Since the Walrus version doesn't use closures, we keep it simple
    return x

def closure_benchmark(n):
    total = 0
    for i in range(n):
        val = make_adder(i)
        total += val + i * 2
    return total

start = time.time()
result = closure_benchmark(50000)
end = time.time()

print(f"Function call stress test (50000 calls)")
print(f"Result: {result}")
print(f"Time: {end - start} seconds")
