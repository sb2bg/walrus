# GC Stress Test - Creates and discards many objects
# Tests garbage collection performance

import time


def create_and_discard(iterations):
    for i in range(iterations):
        # Create lots of temporary lists that get discarded
        temp1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temp2 = [10, 20, 30, 40, 50]
        temp3 = temp1 + temp2
        temp4 = {"a": 1, "b": 2, "c": 3}
        temp5 = [temp1, temp2, temp3]
        # All these become garbage at end of iteration


start = time.time()
create_and_discard(50000)
end = time.time()

print(f"GC stress test (50000 iterations): {end - start} seconds")
