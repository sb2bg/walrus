# While Loop Benchmark - Tests while loop performance vs for
# Pure counting with while

import time

def while_count(n):
    i = 0
    total = 0
    while i < n:
        total += i
        i += 1
    return total

start = time.time()
result = while_count(1000000)
end = time.time()

print(f"While loop sum to 1000000: {result}")
print(f"Time: {end - start} seconds")
