from functools import reduce


def v_add(*a: list) -> list:
    """
    returns addition of 2 vectors
    """

    res = [sum(v) for v in zip(*a)]
    return res


def v_sub(a: list, b: list) -> list:
    """
    returns subtraction of 2 vectors
    """

    res = [v - u for v, u in zip(a, b)]
    return res


def v_smul(s: float, a: list) -> list:
    """
    returns scalar multiplication of 2 vectors
    """

    res = [s * v for v in a]
    return res


def v_hmul(*a: list) -> list:
    """
    returns hadamard product of vectors
    input argument must be 2d matrix
    """

    res = [reduce(lambda n, m: n * m, v) for v in zip(*a)]
    return res


def v_hdiv(a: list, b: list) -> list:
    """
    returns hadamard division of 2 vectors
    input argument must be 2d matrix
    """

    res = [v / u for v, u in zip(a, b)]
    return res


def mat_add(*a: list) -> list:
    """
    returns addition of matrices
    input argument must be 2d matrix
    """

    res = [[sum(v) for v in zip(*i)] for i in zip(*a)]
    return res


def mat_sub(a: list, b: list) -> list:
    """
    returns subtraction of matrix
    input argument must be 2d matrix
    """

    res = [[v - u for v, u in zip(*i)] for i in zip(a, b)]
    return res


def mat_smul(s: float, a: list) -> list:
    """
    returns scalar multiplication of matrix
    input argument must be 2d matrix
    """

    res = [[s * v for v in r] for r in a]
    return res


def mat_hmul(*a: list) -> list:
    """
    returns hadamard product of matrix
    input argument must be 2d matrix
    """

    res = [[reduce(lambda n, m: n * m, v) for v in zip(*i)] for i in zip(*a)]
    return res


def mat_hdiv(a: list, b: list) -> list:
    """
    returns hadamard division of matrix
    input argument must be 2d matrix
    """

    res = [[v / u for v, u in zip(*i)] for i in zip(a, b)]
    return res


def mat_mul(a: list, b: list) -> list:
    """
    returns multiplication of 2 matrices
    input argument must be 2d matrix
    """

    res = [[sum(v * u for v, u in zip(r, c)) for c in zip(*b)] for r in a]
    return res


def mat_mul_all(*a: list) -> list:
    """
    returns multiplication of 2 matrices
    input argument must be 2d matrix
    """

    res = reduce(mat_mul, [*a])
    return res


def mat_tr(a: list) -> float:
    """
    returns trace of matrix
    input argument must be 2d matrix
    """

    res = sum(v[i] for i, v in enumerate([*a]))
    return res

def mat_trans(a: list) -> list:
    """
    returns transposed matrix
    input argument must be 2d matrix
    """

    At = [list(r) for r in zip(*a)]
    return At


def symmetric_check(a: list) -> bool:
    """
    checks whether symmetric matrix or not
    input argument must be 2d matrix
    """

    At = mat_trans(a)
    return a == At


def diag_ele(a: list) -> list:
    """
    returns elements of diagonal matrix
    input argument must be 2d matrix
    """

    d = [v[i] for i, v in enumerate([*a])]
    return d


def mat_diag(a: list) -> list:
    """
    returns diagonal matrix from diagonal elements
    """

    D = [[v if i == j else 0 for j, v in enumerate(r)] for i, r in enumerate(a)]
    return D


def mat_bidiag_u(a: list) -> list:
    """
    transform matrix into upper bidiagonal matrix
    input argument must be 2d matrix
    """

    res = [[0 if i > j or j - i > 1 else v for j, v in enumerate(r)] for i, r in enumerate(a)]
    return res


def mat_bidiag_l(a: list) -> list:
    """
    transform matrix into lower bidiagonal matrix
    input argument must be 2d matrix
    """

    res = [[0 if i < j or i - j > 1 else v for j, v in enumerate(r)] for i, r in enumerate(a)]
    return res


def mat_identity(n: int) -> list:
    """
    returns n by n sized identity matrix
    """

    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return I


def mat_zeros(r: int, c: int) -> list:
    """
    returns r by c sized zero matrix
    """

    Z = [[0 for _ in range(c)] for _ in range(r)]
    return Z


def v_zeros(n: int) -> list:
    """
    returns n sized zero vector
    """

    Z = [0 for _ in range(n)]
    return Z


def mat_tri_u(a: list) -> list:
    """
    transform matrix into upper triangular matrix
    input argument must be 2d matrix
    """

    res = [[0 if i > j else v for j, v in enumerate(r)] for i, r in enumerate(a)]
    return res


def mat_tri_l(a: list) -> list:
    """
    transform matrix into lower triangular matrix
    input argument must be 2d matrix
    """

    res = [[0 if i < j else v for j, v in enumerate(r)] for i, r in enumerate(a)]
    return res


def mat_toeplitz(a: list, b: list) -> list:
    """
    unite 2 lists into toeplitz matrix
    """

    T = [[a[i - j] if i >= j else b[j - i] for j, _ in enumerate(b)] for i, _ in enumerate(a)]
    return T


def v_outer(a: list, b: list) -> list:
    """
    returns outer product, tensor product of 2 vectors
    """

    res = [[v * u for u in b] for v in a]
    return res


def v_inner(a: list, b: list) -> float:
    """
    returns inner product of 2 vectors
    """

    res = sum(v * u for v, u in zip(a, b))
    return res


def householder(v: list) -> list:
    """
    transform vector into householder matrix
    """

    n = len(v)
    V = mat_smul(1 / v_inner(v, v), v_outer(v, v))
    V = mat_smul(2, V)
    H = mat_sub(mat_identity(n), V)
    return H


def determinant(a: list) -> float:
    """
    returns determinant of 2 by 2 matrix
    input argument must be 2d matrix
    """

    det = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])
    return det


def mat_aug_v(a: list, b: list) -> list:
    """
    transform matrix into vector augmented matrix
    input argument must be 2d matrix
    """

    res = [v + [u] for v, u in zip(a, b)]
    return res


def mat_coef(a: list) -> tuple:
    """
    separates coefficient matrix from augmented matrix
    input argument must be 2d matrix
    """

    x = [r[:-1] for r in a]
    y = [v for r in a for v in r[-1:]]
    return x, y


def mat_pivot(mat: list) -> list:
    """
    returns pivoted matrix
    input argument must be 2d matrix
    this is not actual "mathematical" pivoting as this function not selecting rows which first element is 1
    this function just sorts rows as order by descending with first elements of each row
    """

    mat = sorted(mat, key=lambda x: abs(x[0]), reverse=True)
    return mat


def gauss_eli(a: list, b: list) -> list:
    """
    solving equation with Gauss elimination
    input argument must be 2d matrix
    """

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


def gauss_jordan_eli(mat: list) -> list:
    """
    Gauss-Jordan elimination
    transform matrix into Gauss-Jordan eliminated form
    input argument must be 2d matrix
    """

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


def solve_gauss(a: list, b: list) -> list:
    """
    solving equation with Gauss-Jordan elimination
    input argument must be 2d matrix
    """

    mat = mat_aug_v(a, b)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, y = mat_coef(mat)
    return y


def mat_aug_mat(a: list, b: list) -> list:
    """
    transform matrix into matrix augmented matrix
    input argument must be 2d matrix
    """

    res = [v + u for v, u in zip(a, b)]
    return res


def mat_coef_inv(a: list, b: int) -> tuple:
    """
    separates coefficient matrix
    input argument must be 2d matrix
    """

    x = [r[:b] for r in a]
    y = [r[b:] for r in a]
    return x, y


def mat_inv(a: list) -> list:
    """
    returns inverted matrix
    input argument must be 2d matrix
    """

    n = len(a)
    i = mat_identity(n)
    mat = mat_aug_mat(a, i)
    mat = mat_pivot(mat)
    mat = gauss_jordan_eli(mat)
    x, res = mat_coef_inv(mat, n)

    return res


# def det_rec(a: list) -> float:
#     """
#     cofactor expansion
#     """

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


def norm(a: list) -> float:
    """
    returns euclidean norm of vector
    """

    res = sum(i ** 2 for i in a) ** 0.5
    return res


def norm_manhattan(a: list) -> float:
    """
    returns manhattan norm of vector
    """

    res = sum(abs(i) for i in a)
    return res


def cos_similarity(a: list, b: list) -> float:
    """
    returns cosine similarity of 2 vectors
    """

    res = v_inner(a, b) / (norm(a) * norm(b))
    return res


def normalize(a: list) -> list:
    """
    normalize vector
    """

    n = [v / norm(a) for v in a]
    return n


def proj(u: list, v: list) -> list:
    """
    project 'u' vector to 'v' vector
    """

    tmp = v_inner(u, v) / v_inner(v, v)
    res = v_smul(tmp, v)
    return res


def gram_schmidt(s: list) -> list:
    """
    perform Gram-Schmidt Process to matrix
    input argument must be 2d matrix
    """

    res = []
    for i, _ in enumerate(s):
        if i == 0:
            res.append(s[i])
        else:
            tmp = v_sub(s[i], v_add(*[proj(s[i], res[j]) for j in range(i)]))
            res.append(tmp)
    return res


def qr_gramschmidt(a: list) -> tuple:
    """
    QR decomposition/factorization with Gram-Schmidt Process
    input argument must be 2d matrix
    """

    mat = mat_trans(a)
    n = len(mat)
    gs = gram_schmidt(mat)

    q_tmp = [normalize(i) for i in gs]
    q = mat_trans(q_tmp)

    r = [[0 if i > j else v_inner(mat[j], q_tmp[i]) for j in range(n)] for i in range(n)]

    return q, r


# QR decomposition/factorization with householder matrix
def v_sign(a: list) -> int:
    """
    get sign of vector
    returns sign of first element of vector
    """

    res = 1
    if a[0] < 0: res = -1
    return res


def ele_h(a: list) -> list:
    """
    get element of householder matrix except last one
    input argument must be 2d matrix
    """

    at = mat_trans(a)
    nm = norm(at[0])
    e = [1 if j == 0 else 0 for j in range(len(at[0]))]
    sign = v_sign(at[0])
    tmp = v_smul(sign * nm, e)
    v = v_add(at[0], tmp)
    h = householder(v)
    return h


def qr_householder(a: list) -> tuple:
    """
    QR decomposition/factorization with householder matrix
    input argument must be 2d matrix
    """

    n = len(mat_trans(a))
    h_list_tmp = []
    tmp_res = []  # this line is only for evading unbound error, not essential

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

    '''
    interpretation list comprehension of h_list is below

    h_list = []
    for h_tmp in h_list_tmp:
        p = len(h_tmp)

        if p == m:
            tmp = h_tmp
        else:
            tmp = []
            for i in range(m):
                row = []
                for j in range(m):
                    if i < m - p or j < m - p:
                        row.append(I[i][j])
                    else:
                        row.append(h_tmp[i - (m - p)][j - (m - p)])
                tmp.append(row)
        h_list.append(tmp)
    '''

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


def eig_qr(a: list) -> tuple:
    """
    returns eigenvalue and eigenvector by qr decomposition
    input argument must be 2d matrix
    """

    n = len(a)
    v = mat_identity(n)

    for i in range(100):
        q, r = qr_gramschmidt(a)
        a = mat_mul(r, q)
        v = mat_mul(v, q)

    e = diag_ele(a)

    return e, v


def orthogonal_check(a: list) -> bool:
    """
    checks whether orthogonal matrix or not
    input argument must be 2d matrix
    """

    At = mat_trans(a)
    tmp = mat_mul(a, At)
    tmp = mat_smul(1 / tmp[0][0], tmp)  # line for evading floating point error
    I = mat_identity(len(a))

    return tmp == I


def svd(a: list) -> tuple:
    """
    singular value decomposition
    input argument must be 2d matrix
    """

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


def lu_decomp(a: list) -> tuple:
    """
    LU decomposition
    input argument must be 2d matrix
    """

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