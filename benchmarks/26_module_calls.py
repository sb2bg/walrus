# Module Benchmark - imported helper lookups and function calls

from helpers import module_helpers


def build_values():
    values = []
    for i in range(64):
        values.append(i * 3 + 1)
    return values


def module_benchmark(n):
    values = build_values()
    total = 0
    for i in range(n):
        idx = i % len(values)
        total += module_helpers.mix(values[idx], i % 11)
        total += module_helpers.twist(i % 97)
        if i % 8 == 0:
            total += module_helpers.pair_sum(values)
    return total + module_helpers.ANSWER


result = module_benchmark(20000)

print(f"Module call workload: {result}")
