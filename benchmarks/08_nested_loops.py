# Nested Loops Benchmark - Tests deeply nested iteration
# Common pattern in matrix operations

import time

def nested_loop_sum(n):
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += i + j + k
    return total

start = time.time()
result = nested_loop_sum(100)
end = time.time()

print(f"Triple nested loop (100x100x100) sum: {result}")
print(f"Time: {end - start} seconds")
