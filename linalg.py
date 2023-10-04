from functools import reduce

from common import scalar, vector, matrix, production


def v_add(*a: vector) -> vector:
    """returns addition of 2 vectors"""

    return [sum(v) for v in zip(*a)]


def v_sub(a: vector, b: vector) -> vector:
    """returns subtraction of 2 vectors"""

    return [v - u for v, u in zip(a, b)]


def v_smul(s: scalar, a: vector) -> vector:
    """returns scalar multiplication of vector"""

    return [s * v for v in a]


def v_hmul(*a: vector) -> vector:
    """returns hadamard product of vectors"""

    return [production(v) for v in zip(*a)]  # type: ignore


def v_hdiv(a: vector, b: vector) -> vector:
    """returns hadamard division of 2 vectors"""

    return [v / u for v, u in zip(a, b)]


def mat_add(*a: matrix) -> matrix:
    """returns addition of matrices"""

    return [[sum(v) for v in zip(*i)] for i in zip(*a)]


def mat_sub(a: matrix, b: matrix) -> matrix:
    """returns subtraction of matrix"""

    return [[v - u for v, u in zip(*i)] for i in zip(a, b)]


def mat_smul(s: scalar, a: matrix) -> matrix:
    """returns scalar multiplication of matrix"""

    return [[s * v for v in r] for r in a]


def mat_hmul(*a: matrix) -> matrix:
    """returns hadamard product of matrix"""

    return [[production(v) for v in zip(*i)] for i in zip(*a)]  # type: ignore


def mat_hdiv(a: matrix, b: matrix) -> matrix:
    """returns hadamard division of matrix"""

    return [[v / u for v, u in zip(*i)] for i in zip(a, b)]


def mat_mul(a: matrix, b: matrix) -> matrix:
    """returns multiplication of 2 matrices"""

    return [[sum(v * u for v, u in zip(r, c)) for c in zip(*b)] for r in a]


def mat_mul_all(*a: matrix) -> matrix:
    """returns multiplication of given matrices"""

    return reduce(mat_mul, [*a])


def mat_tr(a: matrix) -> scalar:
    """returns trace of matrix"""

    return sum(v[i] for i, v in enumerate([*a]))


def mat_trans(a: matrix) -> matrix:
    """returns transposed matrix"""

    return [list(r) for r in zip(*a)]


def symmetric_check(a: matrix) -> bool:
    """checks whether symmetric matrix or not"""

    return a == mat_trans(a)


def diag_ele(a: matrix) -> vector:
    """returns diagonal elements of matrix"""

    return [v[i] for i, v in enumerate([*a])]


def mat_diag(a: matrix) -> matrix:
    """returns diagonal matrix from matrix"""

    return [[v if i == j else 0 for j, v in enumerate(r)] for i, r in enumerate(a)]


def mat_bidiag_u(a: matrix) -> matrix:
    """transform matrix into upper bidiagonal matrix"""

    return [
        [0 if i > j or j - i > 1 else v for j, v in enumerate(r)]
        for i, r in enumerate(a)
    ]


def mat_bidiag_l(a: matrix) -> matrix:
    """transform matrix into lower bidiagonal matrix"""

    return [
        [0 if i < j or i - j > 1 else v for j, v in enumerate(r)]
        for i, r in enumerate(a)
    ]


def mat_identity(n: int) -> matrix:
    """returns n by n sized identity matrix"""

    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def mat_zeros(r: int, c: int) -> matrix:
    """returns r by c sized zero matrix"""

    return [v_zeros(c) for _ in range(r)]


def v_zeros(n: int) -> vector:
    """returns n sized zero vector"""

    return [0 for _ in range(n)]


def mat_tri_u(a: matrix) -> matrix:
    """transform matrix into upper triangular matrix"""

    return [[0 if i > j else v for j, v in enumerate(r)] for i, r in enumerate(a)]


def mat_tri_l(a: matrix) -> matrix:
    """transform matrix into lower triangular matrix"""

    return [[0 if i < j else v for j, v in enumerate(r)] for i, r in enumerate(a)]


def mat_toeplitz(a: vector, b: vector) -> matrix:
    """unite 2 lists into toeplitz matrix"""

    return [
        [a[i - j] if i >= j else b[j - i] for j, _ in enumerate(b)]
        for i, _ in enumerate(a)
    ]


def v_outer(a: vector, b: vector) -> matrix:
    """returns outer/tensor product of 2 vectors"""

    return [[v * u for u in b] for v in a]


def v_inner(a: vector, b: vector) -> scalar:
    """returns inner product of 2 vectors"""

    return sum(v * u for v, u in zip(a, b))


def householder(v: vector) -> matrix:
    """transform vector into householder matrix"""

    return mat_sub(
        mat_identity(len(v)), mat_smul(2, mat_smul(1 / v_inner(v, v), v_outer(v, v)))
    )


def determinant(a: matrix) -> scalar:
    """returns determinant of 2 by 2 matrix"""

    return (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])


def mat_aug_v(a: matrix, b: vector) -> matrix:
    """transform matrix into vector augmented matrix"""

    return [v + [u] for v, u in zip(a, b)]


def mat_coef(a: matrix) -> tuple:
    """separates coefficient matrix from augmented matrix"""

    x: matrix = [r[:-1] for r in a]
    y: vector = [v for r in a for v in r[-1:]]
    return x, y


def mat_pivot(mat: matrix) -> matrix:
    """
    returns pivoted matrix
    this is not actual "mathematical" pivotingas this function not selecting rows
    which first element is 1
    this function just sorts rows as order by descending with first elements of each row
    """

    return sorted(mat, key=lambda x: abs(x[0]), reverse=True)


def gauss_eli(a: matrix, b: vector) -> vector:
    """solving equation with Gauss elimination"""

    mat: matrix = mat_pivot(mat_aug_v(a, b))
    n: int = len(mat)

    # gauss elimination
    for i in range(n):
        for j in range(i + 1, n):
            tmp = mat[j][i] / mat[i][i]
            for k in range(n + 1):
                mat[j][k] -= tmp * mat[i][k]

    # solve equation
    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            mat[i][n] = mat[i][n] - mat[i][k] * mat[k][n]
        mat[i][n] /= mat[i][i]

    _, y = mat_coef(mat)

    return y


def gauss_jordan_eli(mat: matrix) -> matrix:
    """Gauss-Jordan elimination, transform matrix into Gauss-Jordan eliminated form"""

    n: int = len(mat)

    for i in range(n):
        mat[i] = [ele / mat[i][i] for ele in mat[i]]

        for j in range(n):
            if i == j:
                continue

            mat_tmp = [ele * -mat[j][i] for ele in mat[i]]

            for k in range(len(mat[i])):
                mat[j][k] += mat_tmp[k]

    return mat


def solve_gauss(a: matrix, b: vector) -> vector:
    """solving equation with Gauss-Jordan elimination"""

    _, y = mat_coef(gauss_jordan_eli(mat_pivot(mat_aug_v(a, b))))
    return y


def mat_aug_mat(a: matrix, b: matrix) -> matrix:
    """transform matrix into matrix augmented matrix"""

    return [v + u for v, u in zip(a, b)]


def mat_coef_inv(a: matrix, b: int) -> tuple:
    """separates coefficient matrix"""

    x: matrix = [r[:b] for r in a]
    y: matrix = [r[b:] for r in a]
    return x, y


def mat_inv(a: matrix) -> matrix:
    """returns inverted matrix"""

    n: int = len(a)
    _, res = mat_coef_inv(
        gauss_jordan_eli(mat_pivot(mat_aug_mat(a, mat_identity(n)))), n
    )

    return res


# def det_rec(a: list) -> float:
#     """cofactor expansion"""

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


def norm(a: vector) -> scalar:
    """returns euclidean norm of vector"""

    return sum(i**2 for i in a) ** 0.5


norm_euclidean = norm


def norm_manhattan(a: vector) -> scalar:
    """returns manhattan norm of vector"""

    return sum(abs(i) for i in a)  # type: ignore


def cos_similarity(a: vector, b: vector) -> scalar:
    """returns cosine similarity of 2 vectors"""

    return v_inner(a, b) / (norm(a) * norm(b))


def normalize(a: vector) -> vector:
    """normalize vector"""

    return [v / norm(a) for v in a]


def proj(u: vector, v: vector) -> vector:
    """project 'u' vector to 'v' vector"""

    return v_smul(v_inner(u, v) / v_inner(v, v), v)


def gram_schmidt(s: matrix) -> matrix:
    """perform Gram-Schmidt Process to matrix"""

    res = []
    for i, _ in enumerate(s):
        if i == 0:
            res.append(s[i])
        else:
            res.append(v_sub(s[i], v_add(*[proj(s[i], res[j]) for j in range(i)])))
    return res


def qr_gramschmidt(a: matrix) -> tuple:
    """QR decomposition/factorization with Gram-Schmidt Process"""

    mat: matrix = mat_trans(a)
    n: int = len(mat)
    gs: matrix = gram_schmidt(mat)

    q_tmp: matrix = [normalize(i) for i in gs]
    q: matrix = mat_trans(q_tmp)
    r: matrix = [
        [0 if i > j else v_inner(mat[j], q_tmp[i]) for j in range(n)] for i in range(n)
    ]

    return q, r


# QR decomposition/factorization with householder matrix
def v_sign(a: vector) -> int:
    """get sign of vector(returns sign of first element of vector)"""

    return -1 if a[0] < 0 else 1


def ele_h(a: matrix) -> matrix:
    """get element of householder matrix except last one"""

    at: matrix = mat_trans(a)
    return householder(
        v_add(
            at[0],
            v_smul(
                v_sign(at[0]) * norm(at[0]),
                [1 if j == 0 else 0 for j in range(len(at[0]))],
            ),
        )
    )


def qr_householder(a: matrix) -> tuple:
    """QR decomposition/factorization with householder matrix"""

    n: int = len(mat_trans(a))
    h_list_tmp = []
    tmp_res = []  # this line is only for evading unbound error, not essential

    # get househelder matrixes
    for i in range(n):
        if i == 0:
            res: matrix = ele_h(a)
            h_list_tmp.append(res)
            tmp_res: matrix = mat_mul(res, a)

        elif i < n - 1:
            an = [
                [tmp_res[j][k] for k in range(1, len(tmp_res[0]))]
                for j in range(1, len(tmp_res))
            ]
            res: matrix = ele_h(an)
            h_list_tmp.append(res)
            tmp_res: matrix = mat_mul(res, an)

        else:
            an = [
                tmp_res[j][k]
                for k in range(1, len(tmp_res[0]))
                for j in range(1, len(tmp_res))
            ]
            nm: scalar = norm(an)
            e: vector = [1 if j == 0 else 0 for j in range(len(an))]
            sign: int = v_sign(an)
            tmp = v_smul(sign * nm, e)
            v: vector = v_add(an, tmp)
            h: matrix = householder(v)
            h_list_tmp.append(h)

    # convert househelder matrixes to H_{i} form
    m: int = len(a)
    I: matrix = mat_identity(m)
    h_list = [
        h_tmp
        if len(h_tmp) == m
        else [
            [
                I[i][j]
                if i < m - len(h_tmp) or j < m - len(h_tmp)
                else h_tmp[i - (m - len(h_tmp))][j - (m - len(h_tmp))]
                for i in range(m)
            ]
            for j in range(m)
        ]
        for h_tmp in h_list_tmp
    ]

    """
    interpretation of list comprehension of h_list is below

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
    """

    # calculate Q
    q: matrix = mat_identity(len(h_list[0]))
    for i in h_list:
        q: matrix = mat_mul(q, i)

    # calculate R
    tmp = list(reversed(h_list))
    tmp_i: matrix = mat_identity(len(h_list[0]))
    for i in tmp:
        tmp_i: matrix = mat_mul(tmp_i, i)
    r: matrix = mat_mul(tmp_i, a)

    return q, r


def eig_qr(a: matrix) -> tuple:
    """returns eigenvalue and eigenvector by qr decomposition"""

    n: int = len(a)
    v: matrix = mat_identity(n)

    for _ in range(100):
        q, r = qr_gramschmidt(a)
        a = mat_mul(r, q)
        v: matrix = mat_mul(v, q)

    e: vector = diag_ele(a)

    return e, v


def orthogonal_check(a: matrix) -> bool:
    """checks whether orthogonal matrix or not"""

    tmp: matrix = mat_mul(a, mat_trans(a))
    tmp: matrix = mat_smul(1 / tmp[0][0], tmp)  # line for evading floating point error
    I: matrix = mat_identity(len(a))

    return tmp == I


def svd(a: matrix) -> tuple:
    """singular value decomposition"""

    e, v = eig_qr(mat_mul(mat_trans(a), a))

    s: vector = [i**0.5 for i in e]

    vt: matrix = mat_trans(v)

    ut: matrix = [normalize(v) for v in mat_trans(mat_mul(a, v))]
    u: matrix = mat_trans(ut)

    return u, s, vt


def lu_decomp(a: matrix) -> tuple:
    """LU decomposition"""

    a = mat_pivot(a)
    n: int = len(a)
    m: int = len(a[0])

    l: matrix = mat_zeros(n, m)
    u = []

    for i in range(n):
        u_tmp = a[i]
        val = 1 / u_tmp[i]
        l[i][i] = 1 / val
        u_tmp = [ele * val for ele in u_tmp]
        u.append(u_tmp)

        for j in range(i + 1, n):
            r = a[j]
            a_tmp = [ele * -r[i] for ele in u_tmp]
            l[j][i] = r[i]
            a[j] = [a_tmp[k] + r[k] for k in range(m)]

    return l, u


if __name__ == "__main__":
    a: vector = [1, 2, 3]
    b: vector = [2, 4, 8]
    print(f"\n{a=}\n{b=}\n")

    print(f"addition of vector: {v_add(a, b)}")
    print(f"subtraction of vector: {v_sub(a, b)}")
    print(f"scalar multiplication of vector: {v_smul(3, a)}")
    print(f"hadamard product of vector: {v_hmul(a, b)}")
    print(f"hadamard division of vector: {v_hdiv(a, b)}")

    c: matrix = [[1, 2], [3, 4]]
    d: matrix = [[5, 6], [7, 8]]
    print(f"\n{c=}\n{d=}\n")

    print(f"addition of matrix: {mat_add(c, d)}")
    print(f"subtraction of matrix: {mat_sub(c, d)}")
    print(f"scalar multiplication of matrix: {mat_smul(3, c)}")
    print(f"hadamard product of matrix: {mat_hmul(c, d)}")
    print(f"hadamard division of matrix: {mat_hdiv(c, d)}")
    print(f"multiplication of matrix: {mat_mul(c, d)}")

    e: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"\n{e=}\n")

    print(f"trace of matrix: {mat_tr(e)}")
    print(f"transposed matrix: {mat_trans(e)}")

    e_1: matrix = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    print(f"\n{e_1=}\n")

    print(f"symmetric matrix check: {symmetric_check(e)}")
    print(f"symmetric matrix check: {symmetric_check(e_1)}")
    print(f"diagonal matrix elements: {diag_ele(e)}")
    print(f"diagonal matrix: {mat_diag(e)}")

    f: matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    print(f"\n{f=}\n")

    print(f"upper bidiagonal matrix: {mat_bidiag_u(f)}")
    print(f"lower bidiagonal matrix: {mat_bidiag_l(f)}")

    print(f"identity matrix: {mat_identity(3)}")
    print(f"zero matrix: {mat_zeros(2, 3)}")
    print(f"zero vector: {v_zeros(3)}")
    print(f"upper triangular matrix: {mat_tri_u(f)}")
    print(f"lower triangular matrix: {mat_tri_l(f)}")

    g: vector = [1, 2, 3, 4]
    h: vector = [5, 6, 7, 8]
    print(f"\n{g=}\n{h=}\n")

    print(f"toeplitz matrix: {mat_toeplitz(g, h)}")

    i: vector = [1, 2, 3, 4]
    print(f"\n{i=}\n")

    print(f"v_outer: {v_outer(g, h)}")
    print(f"v_inner: {v_inner(g, h)}")

    print(f"householder matrix: {householder(i)}")

    x: matrix = [[1, 0, -2], [0, 5, 6], [7, 8, 0]]
    y: vector = [4, 5, 3]
    print(f"\n{x=}\n{y=}\n")

    print(f"creating vector augmented matrix: {mat_aug_v(x, y)}")
    print(f"coefficient matrix: {mat_coef(mat_aug_v(x, y))}")
    print(f"sorting augmented matrix: {mat_pivot(mat_aug_v(x, y))}")
    print(f"Gauss elimination: {gauss_eli(x, y)}")
    print(f"Gauss Jordan elimination: {gauss_jordan_eli(mat_aug_v(x, y))}")
    print(f"Solve equation with Gauss-Jordan elimination: {solve_gauss(x, y)}")
    print(f"creating matrix augmented matrix: {mat_aug_mat(x, mat_identity(len(x)))}")
    print(f"inverse matrix: {mat_inv(x)}")

    # j = [[5, 2, -3, 4], [5, 9, 7, 8], [11, 10, 6, 12], [13, 14, 15, 16]]
    # print(f'\n{j=}\n')

    # print(f'det(A): {det_rec(j)}')

    print(f"norm of a: {norm(a)}")
    print(f"cos similarity of a, b: {cos_similarity(a, b)}")
    print(f"normalization of a: {normalize(a)}")
    print(f"projection of a, b: {proj(a, b)}")

    s: matrix = [[1, 0, 1], [0, 1, 1], [1, 2, 0]]
    print(f"\n{s=}\n")

    print(f"gram-schmidt of s: {gram_schmidt(s)}")
    print(f"QR decomposition of s with Gram-Schmidt Process: {qr_gramschmidt(s)}")
    print(f"QR decomposition of s with householder: {qr_householder(s)}")

    k: matrix = [[3, 2, 1], [2, 1, 4], [1, 4, 2]]
    print(f"\n{k=}\n")

    print(f"eigenvalue and eigenvector by qr decomposition: {eig_qr(k)}")

    l: matrix = [[1, 1], [1, -1]]
    print(f"\n{l=}\n")

    print(f"orthogonal_check: {orthogonal_check(l)}")

    m: matrix = [[3, 6], [2, 3], [1, 2], [5, 5]]
    print(f"\n{m=}\n")

    print(f"singular value decomposition: {svd(m)}")
    print(f"lu decomposition of k: {lu_decomp(k)}")
