import copy

# addition of vector
def v_add(a, b):
    n = len(a)

    res = [a[i] + b[i] for i in range(n)]

    return res

# subtraction of vector
def v_sub(a, b):
    n = len(a)

    res = [a[i] - b[i] for i in range(n)]

    return res

# scalar multiplication of vector
def v_smul(s, a):
    n = len(a)

    res = [s * a[i] for i in range(n)]

    return res

# hadamard product of vector
def v_hmul(a, b):
    n = len(a)

    res = [a[i] * b[i] for i in range(n)]

    return res

# hadamard division of vector
def v_hdiv(a, b):
    n = len(a)

    res = [a[i] / b[i] for i in range(n)]

    return res

# addition of matrix
def mat_add(a, b):
    n = len(a)
    m = len(a[0])

    res = [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]

    return res

# subtraction of matrix
def mat_sub(a, b):
    n = len(a)
    m = len(a[0])

    res = [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]

    return res

# scalar multiplication of matrix
def mat_smul(s, a):
    n = len(a)
    m = len(a[0])

    res = [[s * a[i][j] for j in range(m)] for i in range(n)]

    return res

# hadamard product of matrix
def mat_hmul(a, b):
    n = len(a)
    m = len(a[0])

    res = [[a[i][j] * b[i][j] for j in range(m)] for i in range(n)]

    return res

# hadamard division of matrix
def mat_hdiv(a, b):
    n = len(a)
    m = len(a[0])

    res = [[a[i][j] / b[i][j] for j in range(m)] for i in range(n)]

    return res

# multiplication of matrix
def mat_mul(a, b):
    n = len(a)
    m1 = len(a[0])
    m2 = len(b[0])

    res = [[sum(a[i][k] * b[k][j] for k in range(m1)) for j in range(m2)] for i in range(n)]

    return res

# trace of matrix
def mat_tr(a):
    n = len(a)

    res = sum(a[i][i] for i in range(n))

    return res

# transposed matrix
def mat_transpose(a):
    n = len(a)
    m = len(a[0])

    At = [[a[j][i] for j in range(n)] for i in range(m)]

    return At

# symmetric matrix check
def symmetric_check(a):
    At = mat_transpose(a)

    return a == At

# elements of diagonal matrix
def diag_ele(a):
    n = len(a)

    d = [a[i][i] for i in range(n)]

    return d

# diagonal matrix
def mat_diag(a):
    d = diag_ele(a)
    n = len(d)

    D = [[d[i] if i == j else 0 for j in range(n)] for i in range(n)]

    return D

# upper bidiagonal matrix
def mat_bidiag_u(a):
    n = len(a)
    m = len(a[0])

    res = [[0 if i > j or j-i > 1 else a[i][j] for j in range(m)] for i in range(n)]

    return res

# lower bidiagonal matrix
def mat_bidiag_l(a):
    n = len(a)
    m = len(a[0])

    res = [[0 if i < j or i-j > 1 else a[i][j] for j in range(m)] for i in range(n)]

    return res

# identity matrix
def mat_identity(size):

    I = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    return I

# zero matrix
def mat_zeros(row, col):

    Z = [[0 for j in range(col)] for i in range(row)]

    return Z

# upper triangular matrix
def mat_tri_u(a):
    n = len(a)
    m = len(a[0])

    res = [[0 if i > j else a[i][j] for j in range(m)] for i in range(n)]

    return res

# lower triangular matrix
def mat_tri_l(a):
    n = len(a)
    m = len(a[0])

    res = [[0 if i < j else a[i][j] for j in range(m)] for i in range(n)]

    return res

# toeplitz matrix
def mat_toeplitz(a, b):
    n1 = len(a)
    n2 = len(b)

    T = [[a[i-j] if i >= j else b[j-i] for j in range(n2)] for i in range(n1)]

    return T

# outer product of vector
def v_outer(a, b):
    n1 = len(a)
    n2 = len(b)

    res = [[a[i] * b[j] for j in range(n2)] for i in range(n1)]

    return res

# inner product of vector
def v_inner(a, b):
    n = len(a)

    res = sum(a[i] * b[i] for i in range(n))

    return res

# householder matrix
def householder(v):
    n = len(v)
    outer = v_outer(v, v)
    inner = v_inner(v, v)

    V1 = mat_smul(1/inner, outer)
    V2 = mat_smul(2, V1)

    H = mat_sub(mat_identity(n), V2)

    return H

# creating augmented matrix
def mat_aug(a, b):
    x = copy.deepcopy(a)

    x = [x + [b[i]] for i, x in enumerate(x)]

    return x

# separating coefficient matrix
def mat_coef(a):
    n = len(a)

    x = [a[i][:-1] for i in range(n)]
    y = [y for i in range(n) for y in a[i][-1:]]

    return x, y

# pivoting augmented matrix
def mat_pivot(mat):

    mat = sorted(mat, key=lambda x: abs(x[0]), reverse=True)

    return mat

# Gauss elimination
def gauss_eli(a, b):
    mat = mat_aug(a, b)
    mat = mat_pivot(mat)
    n = len(mat)

    # gauss elimination
    for i in range(n):
        for j in range(i+1, n):
            tmp = mat[j][i] / mat[i][i]
            for k in range(n + 1):
                mat[j][k] -= tmp * mat[i][k]

    # solve equation
    for i in range(n-1, -1, -1):
        for k in range(i+1, n):
            mat[i][n] = mat[i][n] - mat[i][k] * mat[k][n]
        mat[i][n] /= mat[i][i]

    x, y = mat_coef(mat)

    return y

# Gauss-Jordan elimination
def gauss_jordan_eli(a):
    mat = copy.deepcopy(a)
    n = len(mat)

    for i in range(n):
        mat[i] = [ele / mat[i][i] for ele in mat[i]]

        for j in range(n):
            if i == j:
                continue

            mat_tmp = [ele * -mat[j][i] for ele in mat[i]]

            for k in range(len(mat[i])):
                mat[j][k] += mat_tmp[k]

    return mat

# solve equation with Gauss-Jordan elimination
def solve(a, b):
    mat = mat_aug(a, b)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, y = mat_coef(mat)

    return y

# creating matrix augmented matrix
def mat_aug_inv(a, b):
    x = copy.deepcopy(a)

    x = [x + b[i] for i, x in enumerate(x)]

    return x

# separating coefficient matrix
def mat_coef_inv(a, b):
    n = len(a)

    x = [a[i][:b] for i in range(n)]
    y = [[y for y in a[i][b:]] for i in range(n)]

    return x, y

# inverse matrix
def mat_inv(a):
    n = len(a)
    i = mat_identity(n)
    mat = mat_aug_inv(a, i)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, res = mat_coef_inv(mat, n)

    return res

# # cofactor expansion
# def det_rec(a):
#     n = len(a)
#     res = 0

#     if n == 2:
#         res = a[0][0] * a[1][1] - a[1][0] * a[0][1]

#         return res

#     else:
#         for i in range(n):
#             x = copy.deepcopy(a)
#             x = x[1:]
#             nx = len(x)

#             for j in range(nx):
#                 x[j] = x[j][0:i] + x[j][i+1:]

#             sign = (-1) ** i
#             res += sign * a[0][i] * det_rec(x)

#         return res

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [2, 4, 8]
    print(f'\na = {a}\nb = {b}')

    print(f'\naddition of vector: {v_add(a, b)}')
    print(f'\nsubtraction of vector: {v_sub(a, b)}')
    print(f'\nscalar multiplication of vector: {v_smul(3, a)}')
    print(f'\nhadamard product of vector: {v_hmul(a, b)}')
    print(f'\nhadamard division of vector: {v_hdiv(a, b)}')

    c = [[1, 2], [3, 4]]
    d = [[5, 6], [7, 8]]
    print(f'\nc = {c}\nd = {d}')

    print(f'\naddition of matrix: {mat_add(c, d)}')
    print(f'\nsubtraction of matrix: {mat_sub(c, d)}')
    print(f'\nscalar multiplication of matrix: {mat_smul(3, c)}')
    print(f'\nhadamard product of matrix: {mat_hmul(c, d)}')
    print(f'\nhadamard division of matrix: {mat_hdiv(c, d)}')
    print(f'\nmultiplication of matrix: {mat_mul(c, d)}')

    e = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f'\ne = {e}')

    print(f'\ntrace of matrix: {mat_tr(e)}')
    print(f'\ntransposed matrix: {mat_transpose(e)}')

    e_1 = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    print(f'\ne_1 = {e_1}')

    print(f'\nsymmetric matrix check: {symmetric_check(e)}')
    print(f'\nsymmetric matrix check: {symmetric_check(e_1)}')
    print(f'\ndiagonal matrix elements: {diag_ele(e)}')
    print(f'\ndiagonal matrix: {mat_diag(e)}')

    f = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    print(f'\nf = {f}')

    print(f'\nupper bidiagonal matrix: {mat_bidiag_u(f)}')
    print(f'\nlower bidiagonal matrix: {mat_bidiag_l(f)}')

    print(f'\nidentity matrix: {mat_identity(3)}')
    print(f'\nzero matrix: {mat_zeros(2, 3)}')
    print(f'\nupper triangular matrix: {mat_tri_u(f)}')
    print(f'\nlower triangular matrix: {mat_tri_l(f)}')

    g = [1, 2, 3, 4]
    h = [5, 6, 7, 8]
    print(f'\ng = {g}\nh = {h}')

    print(f'\ntoeplitz matrix: {mat_toeplitz(g, h)}')

    i = [1, 2, 3, 4]
    print(f'\ni = {i}')

    print(f'\nhouseholder matrix: {householder(i)}')

    x = [[1, 0, -2], [0, 5, 6], [7, 8, 0]]
    y = [4, 5, 3]
    print(f'\nx = {x}\ny = {y}')

    print(f'\ncreating augmented matrix: {mat_aug(x, y)}')
    print(f'\nsorting augmented matrix: {mat_pivot(mat_aug(x, y))}')
    print(f'\nGauss elimination: {gauss_eli(x, y)}')
    print(f'\nGauss Jordan elimination: {gauss_jordan_eli(mat_aug(x, y))}')
    print(f'\nSolve equation with Gauss-Jordan elimination: {solve(x, y)}')
    print(f'\ninverse matrix: {mat_inv(x)}')

    # j = [[5, 2, -3, 4], [5, 9, 7, 8], [11, 10, 6, 12], [13, 14, 15, 16]]
    # print(f'\nj = {j}')

    # print(f'\ndet(A): {det_rec(j)}')