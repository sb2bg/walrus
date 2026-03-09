ANSWER = 42
NAME = "module_helpers"


def mix(a, b):
    return a * 3 + b * 2 - (a % 5)


def twist(x):
    return mix(x, x % 7) + ANSWER


def pair_sum(values):
    total = 0
    for value in values:
        total += twist(value)
    return total
