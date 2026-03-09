# Exception Flow Benchmark - mixed fast path and throw/catch overhead


def maybe_fail(i):
    if i % 7 == 0:
        raise ValueError("skip")
    return i * 3 - 1


def exception_benchmark(n):
    total = 0
    for i in range(n):
        try:
            total += maybe_fail(i)
            if i % 11 == 0:
                total += 2
        except ValueError as err:
            total += len(str(err)) + i
    return total


result = exception_benchmark(50000)

print(f"Exception flow workload: {result}")
