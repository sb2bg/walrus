# Dictionary Operations Benchmark - Tests hash map performance
# Insert, lookup, and iteration

import time


def dict_benchmark(n):
    d = {}

    # Insert n items
    for i in range(n):
        key = f"key_{i}"
        d[key] = i * 2

    # Look up all items
    total = 0
    for i in range(n):
        key = f"key_{i}"
        total += d[key]

    return total


start = time.time()
result = dict_benchmark(10000)
end = time.time()

print(f"Dict operations with 10000 entries: {result}")
print(f"Time: {end - start} seconds")
