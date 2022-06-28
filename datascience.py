# expected value of x
def bar(x):
    res = sum(x) / len(x)

    return res

# covariance
def cov(a, b):
    n = len(a)
    a_bar = bar(a)
    b_bar = bar(b)

    res = sum((a[i] - a_bar) * (b[i] - b_bar) for i in range(n)) / (n - 1)

    return res

# correlation pearson
def pearson(a, b):
    res = cov(a, b) / ((cov(a, a) * cov(b, b)) ** 0.5)

    return res

# linear regression
def lineFit(x, y):
    x_bar = bar(x)
    y_bar = bar(y)

    tmp_0 = [(i - x_bar) * j for i, j in zip(x, y)]
    tmp_1 = [(i - x_bar) * i for i in x]

    w = sum(tmp_0) / sum(tmp_1)
    i = y_bar - (w * x_bar)

    return i, w

if __name__ == "__main__":
    a = [2.23, 4.78, 7.21, 9.37, 11.64, 14.23, 16.55, 18.70, 21.05, 23.21]
    b = [139, 123, 115, 96, 62, 54, 10, -3, -13, -55]
    print(f'a: {a}')
    print(f'b: {b}')

    print(f'\ncov of a, b: {cov(a, b)}')
    print(f'\ncorrelation pearson: {pearson(a, b)}')

    print(f'\nlinear regression of a, b: {lineFit(a, b)}')