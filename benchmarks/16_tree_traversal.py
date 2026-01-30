# Binary Tree Benchmark - Tests recursive data structures
# Build and traverse a binary tree

import time
import sys

sys.setrecursionlimit(50000)


def make_tree(depth):
    if depth <= 0:
        return None
    return {"value": depth, "left": make_tree(depth - 1), "right": make_tree(depth - 1)}


def count_nodes(tree):
    if tree is None:
        return 0
    return 1 + count_nodes(tree["left"]) + count_nodes(tree["right"])


def sum_tree(tree):
    if tree is None:
        return 0
    return tree["value"] + sum_tree(tree["left"]) + sum_tree(tree["right"])


depth = 15
start = time.time()
tree = make_tree(depth)
nodes = count_nodes(tree)
total = sum_tree(tree)
end = time.time()

print(f"Binary tree of depth {depth}")
print(f"Node count: {nodes}")
print(f"Sum of values: {total}")
print(f"Time: {end - start} seconds")
