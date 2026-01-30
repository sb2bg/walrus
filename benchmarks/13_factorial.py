# Factorial Benchmark - Tests recursion and big numbers
# Many recursive calls

import time

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

start = time.time()
total = 0
for i in range(100000):
    total += factorial(12)
end = time.time()

print(f"Computed factorial(12) 100000 times")
print(f"factorial(12) = {factorial(12)}")
print(f"Total sum: {total}")
print(f"Time: {end - start} seconds")
