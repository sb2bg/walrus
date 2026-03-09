# List Slicing Benchmark - repeated slice allocation and window reduction


def build_data(n):
    data = []
    for i in range(n):
        data.append((i * 13 + 7) % 997)
    return data


def slice_benchmark(data, rounds, width):
    total = 0
    limit = len(data) - width - 1
    for r in range(rounds):
        start = (r * 7) % limit
        window = data[start : start + width]
        middle = window[3 : width - 3]
        total += window[0] + window[-1]
        total += middle[0] + middle[-1] + len(middle)
    return total


dataset = build_data(1024)
result = slice_benchmark(dataset, 30000, 24)

print(f"List slicing workload: {result}")
