# Nested Collection Benchmark - dict/list reads, writes, and nested mutation


def build_data(n):
    items = []
    for i in range(n):
        items.append(
            {
                "count": i,
                "values": [i, i + 1, i + 2, i + 3],
                "meta": {"seen": i % 3},
            }
        )
    return items


def nested_benchmark(items, rounds):
    total = 0
    size = len(items)
    for step in range(rounds):
        idx = (step * 7) % size
        entry = items[idx]
        values = entry["values"]
        meta = entry["meta"]
        entry["count"] = entry["count"] + values[0] + values[-1]
        values[1] = values[1] + step % 5
        meta["seen"] = meta["seen"] + 1
        total += entry["count"] + values[1] + meta["seen"]
    return total


data = build_data(512)
result = nested_benchmark(data, 60000)

print(f"Nested collection workload: {result}")
