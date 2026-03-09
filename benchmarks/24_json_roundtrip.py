# JSON Benchmark - repeated decode, mutation, and encode roundtrips

import json


def build_payload():
    users = []
    for i in range(12):
        users.append(
            {
                "id": i,
                "name": f"user_{i}",
                "flags": [i % 2 == 0, i % 3 == 0],
                "score": (i * 17) % 101,
            }
        )

    return {
        "version": 3,
        "name": "walrus",
        "active": True,
        "thresholds": [1, 3, 5, 7, 11],
        "users": users,
        "meta": {"region": "us", "tags": ["vm", "jit", "bench"], "limit": 128},
    }


def json_benchmark(text, rounds):
    total = 0
    for i in range(rounds):
        obj = json.loads(text)
        obj["meta"]["limit"] += i % 5
        encoded = json.dumps(obj, separators=(",", ":"))
        user = obj["users"][i % len(obj["users"])]
        total += len(encoded) + user["score"]
    return total


payload = build_payload()
text = json.dumps(payload, separators=(",", ":"))
result = json_benchmark(text, 4000)

print(f"JSON roundtrip workload: {result}")
