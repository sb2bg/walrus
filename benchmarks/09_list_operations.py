# List Operations Benchmark - Tests list manipulation
# Append, access, and modify operations

import time

def list_benchmark(n):
    # Build a list
    arr = []
    for i in range(n):
        arr.append(i)
    
    # Sum all elements
    total = 0
    for x in arr:
        total += x
    
    # Access elements randomly-ish
    access_sum = 0
    for i in range(n):
        idx = (i * 7 + 13) % n
        access_sum += arr[idx]
    
    return total + access_sum

start = time.time()
result = list_benchmark(10000)
end = time.time()

print(f"List operations on 10000 elements: {result}")
print(f"Time: {end - start} seconds")
