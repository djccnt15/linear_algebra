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

# zero vector
def v_zeros(size):

    Z = [0 for i in range(size)]

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

# outer product, tensor product of vector
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

    V1 = mat_smul(1 / inner, outer)
    V2 = mat_smul(2, V1)

    H = mat_sub(mat_identity(n), V2)

    return H

# determinant of 2 by 2 matrix
def determinant(a):
    det = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])

    return det

a = [[1, 2], [3, 4]]

# creating vector augmented matrix
def mat_aug_v(a, b):
    n = len(a)

    res = [a[i] + [b[i]] for i in range(n)]

    return res

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
    mat = mat_aug_v(a, b)
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
def solve(a, b):
    mat = mat_aug_v(a, b)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, y = mat_coef(mat)

    return y

# creating matrix augmented matrix
def mat_aug_mat(a, b):
    n = len(a)

    res = [a[i] + b[i] for i in range(n)]

    return res

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
    n = len(a)
    res = sum(a[i] ** 2 for i in range(n)) ** 0.5

    return res

# cosine similarity
def cos_similarity(a, b):
    inner = v_inner(a, b)
    nm_a = norm(a)
    nm_b = norm(b)

    res = inner / (nm_a * nm_b)

    return res

# normalize vector
def normalize(a):
    n = len(a)
    v = [a[i] / norm(a) for i in range(n)]

    return v

# projection
def proj(a, b):
    tmp = v_inner(a, b) / v_inner(b, b)
    res = v_smul(tmp, b)

    return res

# Gram-Schmidt Process
def gram_schmidt(s):
    n = len(s)
    m = len(s[0])
    res = []

    for i in range(n):
        if i == 0:
            res.append(s[i])

        else:
            tmp_list = [proj(s[i], res[j]) for j in range(i)]

            tmp = v_zeros(m)
            for k in range(len(tmp_list)):
                tmp = v_add(tmp, tmp_list[k])
            tmp = v_sub(s[i], tmp)
            res.append(tmp)

    return res

# QR decomposition, QR factorization with Gram-Schmidt Process
def qr_gramschmidt(a):
    mat = mat_transpose(a)
    n = len(mat)
    tmp = gram_schmidt(mat)

    q_tmp = [normalize(i) for i in tmp]
    q = mat_transpose(q_tmp)

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
    at = mat_transpose(a)
    nm = norm(at[0])
    e = [1 if j == 0 else 0 for j in range(len(at[0]))]
    sign = v_sign(at[0])
    tmp = v_smul(sign * nm, e)
    v = v_add(at[0], tmp)
    h = householder(v)

    return h

# QR decomposition
def qr_householder(a):
    n = len(mat_transpose(a))
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
    print(f'\nzero vector: {v_zeros(3)}')
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

    print(f'\ncreating augmented matrix: {mat_aug_v(x, y)}')
    print(f'\nsorting augmented matrix: {mat_pivot(mat_aug_v(x, y))}')
    print(f'\nGauss elimination: {gauss_eli(x, y)}')
    print(f'\nGauss Jordan elimination: {gauss_jordan_eli(mat_aug_v(x, y))}')
    print(f'\nSolve equation with Gauss-Jordan elimination: {solve(x, y)}')
    print(f'\ninverse matrix: {mat_inv(x)}')

    # j = [[5, 2, -3, 4], [5, 9, 7, 8], [11, 10, 6, 12], [13, 14, 15, 16]]
    # print(f'\nj = {j}')

    # print(f'\ndet(A): {det_rec(j)}')

    print(f'\nnorm of a: {norm(a)}')
    print(f'\ncos similarity of a, b: {cos_similarity(a, b)}')
    print(f'\nnormalization of a: {normalize(a)}')
    print(f'\nprojection of a, b: {proj(a, b)}')

    s = [[10, -10, 4, 10], [20, 4, -20, 8], [30, 40, 2, 6], [10, -10, 0, 3]]
    print(f'\ns = {s}')

    print(f'\ngram-schmidt of s: {gram_schmidt(s)}')
    print(f'\nQR decomposition of s with Gram-Schmidt Process: {qr_gramschmidt(s)}')
    print(f'\nQR decomposition of s with householder: {qr_householder(s)}')