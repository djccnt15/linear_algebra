from functools import reduce

# addition of vector
def v_add(*a):
    res = [sum(v) for v in zip(*a)]

    return res

# subtraction of vector
def v_sub(a, b):
    res = [v - u for v, u in zip(a, b)]

    return res

# scalar multiplication of vector
def v_smul(s, a):
    res = [s * v for v in a]

    return res

# hadamard product of vector
def v_hmul(*a):
    res = [reduce(lambda n, m: n * m, v) for v in zip(*a)]

    return res

# hadamard division of vector
def v_hdiv(a, b):
    res = [v / u for v, u in zip(a, b)]

    return res

# addition of matrix
def mat_add(*a):
    res = [[sum(v) for v in zip(*i)] for i in zip(*a)]

    return res

# subtraction of matrix
def mat_sub(a, b):
    res = [[v - u for v, u in zip(*i)] for i in zip(a, b)]

    return res

# scalar multiplication of matrix
def mat_smul(s, a):
    res = [[s * v for v in r] for r in a]

    return res

# hadamard product of matrix
def mat_hmul(*a):
    res = [[reduce(lambda n, m: n * m, v) for v in zip(*i)] for i in zip(*a)]

    return res

# hadamard division of matrix
def mat_hdiv(a, b):
    res = [[v / u for v, u in zip(*i)] for i in zip(a, b)]

    return res

# multiplication of matrix
def mat_mul(a, b):
    res = [[sum(v * u for v, u in zip(r, c)) for c in zip(*b)] for r in a]

    return res

def mat_mul_all(*a):
    res = reduce(mat_mul, [*a])

    return res

# trace of matrix
def mat_tr(a):
    res = sum(v[i] for i, v in enumerate([*a]))

    return res

# transposed matrix
def mat_trans(a):
    At = [list(r) for r in zip(*a)]

    return At

# symmetric matrix check
def symmetric_check(a):
    At = mat_trans(a)

    return a == At

# elements of diagonal matrix
def diag_ele(a):
    d = [v[i] for i, v in enumerate([*a])]

    return d

# diagonal matrix
def mat_diag(a):
    D = [[v if i == j else 0 for j, v in enumerate(r)] for i, r in enumerate(a)]

    return D

# upper bidiagonal matrix
def mat_bidiag_u(a):
    res = [[0 if i > j or j - i > 1 else v for j, v in enumerate(r)] for i, r in enumerate(a)]

    return res

# lower bidiagonal matrix
def mat_bidiag_l(a):
    res = [[0 if i < j or i - j > 1 else v for j, v in enumerate(r)] for i, r in enumerate(a)]

    return res

# identity matrix
def mat_identity(n):
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    return I

# zero matrix
def mat_zeros(r, c):
    Z = [[0 for _ in range(c)] for _ in range(r)]

    return Z

# zero vector
def v_zeros(n):
    Z = [0 for i in range(n)]

    return Z

# upper triangular matrix
def mat_tri_u(a):
    res = [[0 if i > j else v for j, v in enumerate(r)] for i, r in enumerate(a)]

    return res

# lower triangular matrix
def mat_tri_l(a):
    res = [[0 if i < j else v for j, v in enumerate(r)] for i, r in enumerate(a)]

    return res

# toeplitz matrix
def mat_toeplitz(a, b):
    T = [[a[i - j] if i >= j else b[j - i] for j, _ in enumerate(b)] for i, _ in enumerate(a)]

    return T

# outer product, tensor product of vector
def v_outer(a, b):
    res = [[v * u for u in b] for v in a]

    return res

# inner product of vector
def v_inner(a, b):
    res = sum(v * u for v, u in zip(a, b))

    return res

# householder matrix
def householder(v):
    n = len(v)
    V = mat_smul(1 / v_inner(v, v), v_outer(v, v))
    V = mat_smul(2, V)

    H = mat_sub(mat_identity(n), V)

    return H

# determinant of 2 by 2 matrix
def determinant(a):
    det = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])

    return det

# creating vector augmented matrix
def mat_aug_v(a, b):
    res = [v + [u] for v, u in zip(a, b)]

    return res

# separating coefficient matrix
def mat_coef(a):
    x = [r[:-1] for r in a]
    y = [v for r in a for v in r[-1:]]

    return x, y

# pivoting augmented matrix
def mat_pivot(mat):
    mat = sorted(mat, key=lambda x: abs(x[0]), reverse=True)

    return mat

# Gauss elimination
def gauss_eli(a, b):
    mat = mat_aug_v(a, b)
    mat = mat_pivot(mat)
    n = len(mat)

    # gauss elimination
    for i in range(n):
        for j in range(i+1, n):
            tmp = mat[j][i] / mat[i][i]
            for k in range(n+1):
                mat[j][k] -= tmp * mat[i][k]

    # solve equation
    for i in range(n-1, -1, -1):
        for k in range(i+1, n):
            mat[i][n] = mat[i][n] - mat[i][k] * mat[k][n]
        mat[i][n] /= mat[i][i]

    x, y = mat_coef(mat)

    return y

# Gauss-Jordan elimination
def gauss_jordan_eli(mat):
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
def solve_gauss(a, b):
    mat = mat_aug_v(a, b)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, y = mat_coef(mat)

    return y

# creating matrix augmented matrix
def mat_aug_mat(a, b):
    res = [v + u for v, u in zip(a, b)]

    return res

# separating coefficient matrix
def mat_coef_inv(a, b):
    x = [r[:b] for r in a]
    y = [r[b:] for r in a]

    return x, y

# inverse matrix
def mat_inv(a):
    n = len(a)
    i = mat_identity(n)
    mat = mat_aug_mat(a, i)
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

# norm of vector
def norm(a):
    res = sum(i ** 2 for i in a) ** 0.5

    return res

# cosine similarity
def cos_similarity(a, b):
    res = v_inner(a, b) / (norm(a) * norm(b))

    return res

# normalize vector
def normalize(a):
    n = [v / norm(a) for v in a]

    return n

# projection
def proj(a, b):
    tmp = v_inner(a, b) / v_inner(b, b)
    res = v_smul(tmp, b)

    return res

# Gram-Schmidt Process
def gram_schmidt(s):
    res = []

    for i, _ in enumerate(s):
        if i == 0:
            res.append(s[i])

        else:
            tmp = v_sub(s[i], v_add(*[proj(s[i], res[j]) for j in range(i)]))
            res.append(tmp)

    return res

# QR decomposition, QR factorization with Gram-Schmidt Process
def qr_gramschmidt(a):
    mat = mat_trans(a)
    n = len(mat)
    gs = gram_schmidt(mat)

    q_tmp = [normalize(i) for i in gs]
    q = mat_trans(q_tmp)

    r = [[0 if i > j else v_inner(mat[j], q_tmp[i]) for j in range(n)] for i in range(n)]

    return q, r

# QR decomposition, QR factorization with householder matrix
# sign of vector
def v_sign(a):
    res = 1
    if a[0] < 0: res = -1

    return res

# get element of househelder matrixes except last one
def ele_h(a):
    at = mat_trans(a)
    nm = norm(at[0])
    e = [1 if j == 0 else 0 for j in range(len(at[0]))]
    sign = v_sign(at[0])
    tmp = v_smul(sign * nm, e)
    v = v_add(at[0], tmp)
    h = householder(v)

    return h

# QR decomposition
def qr_householder(a):
    n = len(mat_trans(a))
    h_list_tmp = []

    # get househelder matrixes
    for i in range(n):
        if i == 0:
            res = ele_h(a)
            h_list_tmp.append(res)
            tmp_res = mat_mul(res, a)

        elif i < n - 1:
            an = [[tmp_res[j][k] for k in range(1, len(tmp_res[0]))] for j in range(1, len(tmp_res))]
            res = ele_h(an)
            h_list_tmp.append(res)
            tmp_res = mat_mul(res, an)

        else:
            an = [tmp_res[j][k] for k in range(1, len(tmp_res[0])) for j in range(1, len(tmp_res))]
            nm = norm(an)
            e = [1 if j == 0 else 0 for j in range(len(an))]
            sign = v_sign(an)
            tmp = v_smul(sign * nm, e)
            v = v_add(an, tmp)
            h = householder(v)
            h_list_tmp.append(h)

    # convert househelder matrixes to H_{i} form
    m = len(a)
    I = mat_identity(m)
    h_list = [h_tmp if len(h_tmp) == m \
        else [[I[i][j] if i < m - len(h_tmp) or j < m - len(h_tmp) \
            else h_tmp[i - (m - len(h_tmp))][j - (m - len(h_tmp))] \
                for i in range(m)] for j in range(m)] for h_tmp in h_list_tmp]

    # h_list = []
    # for h_tmp in h_list_tmp:
    #     p = len(h_tmp)

    #     if p == m:
    #         tmp = h_tmp
    #     else:
    #         tmp = []
    #         for i in range(m):
    #             row = []
    #             for j in range(m):
    #                 if i < m - p or j < m - p:
    #                     row.append(I[i][j])
    #                 else:
    #                     row.append(h_tmp[i - (m - p)][j - (m - p)])
    #             tmp.append(row)
    #     h_list.append(tmp)

    # calculate Q
    q = mat_identity(len(h_list[0]))
    for i in h_list:
        q = mat_mul(q, i)

    # calculate R
    tmp = list(reversed(h_list))
    tmp_i = mat_identity(len(h_list[0]))
    for i in tmp:
        tmp_i = mat_mul(tmp_i, i)
    r = mat_mul(tmp_i, a)

    return q, r

# eigenvalue and eigenvector by qr decomposition
def eig_qr(a):
    n = len(a)
    v = mat_identity(n)

    for i in range(100):
        q, r = qr_gramschmidt(a)
        a = mat_mul(r, q)
        v = mat_mul(v, q)

    e = diag_ele(a)

    return e, v

# orthogonal matrix check
def orthogonal_check(a):
    At = mat_trans(a)
    tmp = mat_mul(a, At)
    tmp = mat_smul(1 / tmp[0][0], tmp)  # line for evading floating point error
    I = mat_identity(len(a))

    return tmp == I

# singular value decomposition
def svd(a):
    at = mat_trans(a)
    ata = mat_mul(at, a)
    e, v = eig_qr(ata)

    s = [i ** 0.5 for i in e]

    vt = mat_trans(v)

    av = mat_mul(a, v)
    avt = mat_trans(av)
    ut = [normalize(v) for v in avt]

    u = mat_trans(ut)

    return u, s, vt

# LU decomposition
def lu_decomp(a):
    a = mat_pivot(a)
    n = len(a)
    m = len(a[0])

    l = mat_zeros(n, m)
    u = []

    for i in range(n):
        u_tmp = a[i]
        val = 1 / u_tmp[i]
        l[i][i] = 1 / val
        u_tmp = [ele * val for ele in u_tmp]
        u.append(u_tmp)

        for j in range(i+1, n):
            r = a[j]
            a_tmp = [ele * -r[i] for ele in u_tmp]
            l[j][i] = r[i]
            a[j] = [a_tmp[k] + r[k] for k in range(m)]

    return l, u

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
    print(f'\ntransposed matrix: {mat_trans(e)}')

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
    print(f'\nzero vector: {v_zeros(3)}')
    print(f'\nupper triangular matrix: {mat_tri_u(f)}')
    print(f'\nlower triangular matrix: {mat_tri_l(f)}')

    g = [1, 2, 3, 4]
    h = [5, 6, 7, 8]
    print(f'\ng = {g}\nh = {h}')

    print(f'\ntoeplitz matrix: {mat_toeplitz(g, h)}')

    i = [1, 2, 3, 4]
    print(f'\ni = {i}')

    print(f'\nv_outer = {v_outer(g, h)}')
    print(f'\nv_inner = {v_inner(g, h)}')

    print(f'\nhouseholder matrix: {householder(i)}')

    x = [[1, 0, -2], [0, 5, 6], [7, 8, 0]]
    y = [4, 5, 3]
    print(f'\nx = {x}\ny = {y}')

    print(f'\ncreating vector augmented matrix: {mat_aug_v(x, y)}')
    print(f'\ncoefficient matrix: {mat_coef(mat_aug_v(x, y))}')
    print(f'\nsorting augmented matrix: {mat_pivot(mat_aug_v(x, y))}')
    print(f'\nGauss elimination: {gauss_eli(x, y)}')
    print(f'\nGauss Jordan elimination: {gauss_jordan_eli(mat_aug_v(x, y))}')
    print(f'\nSolve equation with Gauss-Jordan elimination: {solve_gauss(x, y)}')
    print(f'\ncreating matrix augmented matrix: {mat_aug_mat(x, mat_identity(len(x)))}')
    print(f'\ninverse matrix: {mat_inv(x)}')

    # j = [[5, 2, -3, 4], [5, 9, 7, 8], [11, 10, 6, 12], [13, 14, 15, 16]]
    # print(f'\nj = {j}')

    # print(f'\ndet(A): {det_rec(j)}')

    print(f'\nnorm of a: {norm(a)}')
    print(f'\ncos similarity of a, b: {cos_similarity(a, b)}')
    print(f'\nnormalization of a: {normalize(a)}')
    print(f'\nprojection of a, b: {proj(a, b)}')

    s = [[1, 0, 1], [0, 1, 1], [1, 2, 0]]
    print(f'\ns = {s}')

    print(f'\ngram-schmidt of s: {gram_schmidt(s)}')
    print(f'\nQR decomposition of s with Gram-Schmidt Process: {qr_gramschmidt(s)}')
    print(f'\nQR decomposition of s with householder: {qr_householder(s)}')

    k = [[3, 2, 1], [2, 1, 4], [1, 4, 2]]
    print(f'\nk = {k}')

    print(f'\neigenvalue and eigenvector by qr decomposition: {eig_qr(k)}')

    l = [[1, 1], [1, -1]]
    print(f'\nl = {l}')

    print(f'\northogonal_check: {orthogonal_check(l)}')

    m = [[3, 6], [2, 3], [1, 2], [5, 5]]
    print(f'm = {m}')

    print(f'singular value decomposition: {svd(m)}')
    print(f'lu decomposition of k: {lu_decomp(k)}')