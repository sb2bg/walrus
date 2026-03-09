# Async Benchmark - task creation, cooperative yields, and gather

import asyncio


async def worker(seed, rounds):
    acc = seed
    for i in range(rounds):
        acc += (i + seed) % 7
        if i % 4 == 0:
            await asyncio.sleep(0)
    return acc


async def scheduler_benchmark(task_count, rounds):
    tasks = []
    for i in range(task_count):
        tasks.append(asyncio.create_task(worker(i + 1, rounds)))
    results = await asyncio.gather(*tasks)
    total = 0
    for value in results:
        total += value
    return total


result = asyncio.run(scheduler_benchmark(48, 300))

print(f"Async scheduler workload: {result}")
