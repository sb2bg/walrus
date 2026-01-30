# GC Linked Structures - Creates linked list-like structures
# Tests GC handling of nested references

import time


def create_chain(depth):
    if depth <= 0:
        return {"value": 0, "next": None}
    return {"value": depth, "next": create_chain(depth - 1)}


def sum_chain(node):
    total = 0
    while node is not None:
        total += node["value"]
        node = node["next"]
    return total


start = time.time()
total = 0
for i in range(1000):
    chain = create_chain(100)
    total += sum_chain(chain)
end = time.time()

print(f"Created and traversed 1000 chains of depth 100")
print(f"Total sum: {total}")
print(f"Time: {end - start} seconds")
