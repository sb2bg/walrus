# Arithmetic Heavy Benchmark - Tests raw computation speed
# Lots of math operations

import time

def compute(n):
    a = 1
    b = 2
    c = 3
    result = 0
    
    for i in range(n):
        result += (a * b + c) / 2
        a = (a + 1) % 100
        b = (b * 2) % 100
        c = (c + 3) % 100
    
    return result

start = time.time()
result = compute(1000000)
end = time.time()

print(f"Arithmetic operations (1000000 iterations): {result}")
print(f"Time: {end - start} seconds")
