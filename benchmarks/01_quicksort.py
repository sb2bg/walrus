# Quicksort Benchmark - Tests sorting performance
# Sorts a list of 5000 random-ish numbers

import time

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = []
    middle = []
    right = []
    
    for x in arr:
        if x < pivot:
            left.append(x)
        elif x > pivot:
            right.append(x)
        else:
            middle.append(x)
    
    return quicksort(left) + middle + quicksort(right)

# Generate pseudo-random numbers using LCG
def generate_random_list(n, seed):
    result = []
    current = seed
    for i in range(n):
        current = (current * 1103515245 + 12345) % 2147483648
        result.append(current % 10000)
    return result

start = time.time()
data = generate_random_list(5000, 42)
sorted_data = quicksort(data)
end = time.time()

print(f"Quicksort 5000 elements: {end - start} seconds")
print(f"First 5 sorted: {sorted_data[0]}, {sorted_data[1]}, {sorted_data[2]}, {sorted_data[3]}, {sorted_data[4]}")
