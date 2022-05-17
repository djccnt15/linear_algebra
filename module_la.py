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

# addition of matrix
def m_add(a, b):
    n = len(a)
    p = len(a[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p):
            val = a[i][j] + b[i][j]
            row.append(val)
        res.append(row)

    return res

# subtraction of matrix
def m_sub(a, b):
    n = len(a)
    p = len(a[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p):
            val = a[i][j] - b[i][j]
            row.append(val)
        res.append(row)

    return res

# scalar multiplication of matrix
def mat_mul_scaler(b, a):
    n = len(a)
    p = len(a[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p):
            val = b * a[i][j]
            row.append(val)
        res.append(row)

    return res

# hadamard product of matrix
def m_hmul(a, b):
    n = len(a)
    p = len(a[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p):
            val = a[i][j] * b[i][j]
            row.append(val)
        res.append(row)

    return res

# hadamard division of matrix
def m_hdiv(a, b):
    n = len(a)
    p = len(a[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p):
            val = a[i][j] / b[i][j]
            row.append(val)
        res.append(row)

    return res

# multiplication of matrix
def matmul(a, b):
    n = len(a)
    p1 = len(a[0])
    p2 = len(b[0])

    res = []
    for i in range(n):
        row = []
        for j in range(p2):
            val = 0
            for k in range(p1):
                val += a[i][k] * b[k][j]
            row.append(val)
        res.append(row)

    return res

# trace of matrix
def tr(a):
    n = len(a)

    val = 0
    for i in range(n):
        val += a[i][i]

    return val

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [2, 4, 8]
    c = [[1, 2], [3, 4]]
    d = [[5, 6], [7, 8]]
    e = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    print(f'a = {a}\nb = {b}')
    print(f'addition of vector: {v_add(a, b)}')
    print(f'subtraction of vector: {v_sub(a, b)}')
    print(f'scalar multiplication of vector: {v_mul_scaler(3, a)}')
    print(f'hadamard product of vector: {v_hmul(a, b)}')
    print(f'hadamard division of vector: {v_hdiv(a, b)}')

    print(f'c = {c}\nd = {d}')
    print(f'addition of matrix: {m_add(c, d)}')
    print(f'subtraction of matrix: {m_sub(c, d)}')
    print(f'scalar multiplication of matrix: {mat_mul_scaler(3, c)}')
    print(f'hadamard product of matrix: {m_hmul(c, d)}')
    print(f'hadamard division of matrix: {m_hdiv(c, d)}')
    print(f'multiplication of matrix: {matmul(c, d)}')

    print(f'e = {e}')
    print(f'trace of matrix: {tr(e)}')