# String Concatenation Benchmark - Tests string operations
# Building strings through concatenation

import time


def string_benchmark(n):
    result = ""
    for i in range(n):
        result = result + "a"
    return len(result)


start = time.time()
length = string_benchmark(10000)
end = time.time()

print(f"Built string of length {length}")
print(f"Time: {end - start} seconds")
