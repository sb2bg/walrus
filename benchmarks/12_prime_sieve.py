# Sieve of Eratosthenes - Classic algorithm benchmark
# Tests list operations and conditionals

import time


def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False

    p = 2
    while p * p <= n:
        if is_prime[p]:
            multiple = p * p
            while multiple <= n:
                is_prime[multiple] = False
                multiple += p
        p += 1

    primes = []
    for i in range(n + 1):
        if is_prime[i]:
            primes.append(i)

    return primes


start = time.time()
primes = sieve(50000)
end = time.time()

print(f"Found {len(primes)} primes up to 50000")
print(
    f"Last 5 primes: {primes[-5]}, {primes[-4]}, {primes[-3]}, {primes[-2]}, {primes[-1]}"
)
print(f"Time: {end - start} seconds")
