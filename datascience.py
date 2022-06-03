# expected value of x
def x_bar(x):
    res = sum(x) / len(x)

    return res

# covariance
def cov(a, b):
    n = len(a)
    a_bar = x_bar(a)
    b_bar = x_bar(b)

    res = sum((a[i] - a_bar) * (b[i] - b_bar) for i in range(n)) / (n - 1)

    return res

# correlation pearson
def pearson(a, b):

    res = cov(a, b) / ((cov(a, a) * cov(b, b)) ** 0.5)

    return res

if __name__ == "__main__":
    a = [2.23, 4.78, 7.21, 9.37, 11.64, 14.23, 16.55, 18.70, 21.05, 23.21]
    b = [139, 123, 115, 96, 62, 54, 10, -3, -138, -550]
    print(f'a: {a}')
    print(f'b: {b}')

    print(f'\ncov of a, b: {cov(a, b)}')
    print(f'\ncorrelation pearson: {pearson(a, b)}')