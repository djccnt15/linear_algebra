scalar = int | float
vector = list[scalar]
matrix = list[vector]


def production(data: vector) -> float:
    """product all elements in data with for loop"""

    res = 1
    for i in data:
        res *= i
    return res


def production_rec(data: vector) -> float:
    """product all elements in data with recursion"""

    return data[0] if len(data) == 1 else data[0] * production_rec(data[1:])


def factorial(n: int) -> int:
    """returns factorial of number with for loop"""

    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


def factorial_rec(n: int) -> int:
    """returns factorial of number with recursion"""

    return 1 if n <= 1 else n * factorial_rec(n - 1)
