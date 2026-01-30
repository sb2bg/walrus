# Iteration Heavy Benchmark - Pure loop performance
# Tests how fast loops can execute

import time

def count_iterations():
    count = 0
    for i in range(1000000):
        count += 1
    return count

start = time.time()
result = count_iterations()
end = time.time()

print(f"Counted to {result}")
print(f"Time: {end - start} seconds")
