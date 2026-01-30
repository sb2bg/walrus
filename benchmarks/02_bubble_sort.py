# Bubble Sort Benchmark - Tests O(n²) sorting with lots of swaps
# Intentionally inefficient to stress test loops

import time


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def generate_random_list(n, seed):
    result = []
    current = seed
    for i in range(n):
        current = (current * 1103515245 + 12345) % 2147483648
        result.append(current % 10000)
    return result


start = time.time()
data = generate_random_list(1000, 42)
sorted_data = bubble_sort(data)
end = time.time()

print(f"Bubble sort 1000 elements: {end - start} seconds")
print(
    f"First 5 sorted: {sorted_data[0]}, {sorted_data[1]}, {sorted_data[2]}, {sorted_data[3]}, {sorted_data[4]}"
)
