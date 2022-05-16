# addition of vector
def v_add(a, b):
    n = len(a)
    res = []

    for i in range(n):
        val = a[i] + b[i]
        res.append(val)

    return res

# subtraction of vector
def v_sub(a, b):
    n = len(a)
    res = []

    for i in range(n):
        val = a[i] - b[i]
        res.append(val)

    return res

# scalar multiplication of vector
def v_mul_scaler(n, a):
    n = len(a)
    res = []

    for i in range(n):
        val = n * a[i]
        res.append(val)

    return res

# hadamard product of vector
def v_hmul(a, b):
    n = len(a)
    res = []

    for i in range(n):
        val = a[i] * b[i]
        res.append(val)

    return res

# hadamard division of vector
def v_hdiv(a, b):
    n = len(a)
    res = []

    for i in range(n):
        val = a[i] / b[i]
        res.append(val)

    return res

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [2, 4, 8]

    print(f'addition of vector: {v_add(a, b)}')
    print(f'subtration of vector: {v_sub(a, b)}')
    print(f'scalar multiplication of vector: {v_mul_scaler(3, a)}')
    print(f'hadamard product of vector: {v_hmul(a, b)}')
    print(f'hadamard division of vector: {v_hdiv(a, b)}')