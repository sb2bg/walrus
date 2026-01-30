# Ackermann Function Benchmark - Extreme recursion stress test
# Warning: This grows VERY fast, keep parameters small!

import time
import sys

sys.setrecursionlimit(10000)


def ackermann(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ackermann(m - 1, 1)
    return ackermann(m - 1, ackermann(m, n - 1))


start = time.time()
result = ackermann(3, 7)
end = time.time()

print(f"Ackermann(3, 7) = {result}")
print(f"Time: {end - start} seconds")
