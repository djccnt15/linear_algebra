import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 4, 8])

v_add = a + b
v_sub = a - b
v_mul_scaler = 3 * a
v_hmul = a * b
v_hdiv = a / b

# print(a, b)
# print(v_hmul)

c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

m_add = c + d
m_sub = c - d
mat_mul_scaler = 3 * c
m_hmul = np.multiply(c, d)
m_hdiv = c / d
matmul = np.matmul(c, d)
trace = np.trace(e)

# print(c)
# print(d)
# print(m_hmul)
# print(matmul)
# print(trace)