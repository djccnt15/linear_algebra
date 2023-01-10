import math

numeric = list[int | float]


def freq(data: numeric) -> dict:
    """returns frequency of each value"""

    res = {val: data.count(val) for val in set(data)}
    return res


def freq_rel(data: numeric) -> dict:
    """returns relative frequency of each value"""

    res = {val: data.count(val) / len(data) for val in set(data)}
    return res


def freq_cum(data: numeric) -> dict:
    """returns cumulative frequency of sorted value"""

    data_sort = sorted(list(set(data)))
    res = {val: sum(data.count(j) for j in data_sort[:i+1]) for i, val in enumerate(data_sort)}
    return res


def bar(data: numeric) -> float:
    """returns expectation/sample mean"""

    res = sum(data) / len(data)
    return res


def devi(data: numeric) -> dict:
    """returns deviation of each value"""

    b = bar(data)
    res = {val: val - b for val in sorted(list(set(data)))}
    return res


def mean_weight(data: numeric, weights: numeric) -> float:
    """returns weighted mean"""

    res = sum(v * w for v, w in zip(data, weights)) / sum(weights)
    return res


def production(data: numeric) -> float:
    """product all elements in data with for loop"""

    res = 1
    for i in data:
        res *= i
    return res


def production_rec(data: numeric) -> float:
    """product all elements in data with recursion"""

    return data[0] if len(data) == 1 else data[0] * production_rec(data[1:])


def mean_geom(data: numeric) -> float:
    """returns geometric mean of data"""

    res = production(data) ** (1 / len(data))
    return res


def mean_harm(data: numeric) -> float:
    """returns harmonic mean of data"""

    res = len(data) / (sum(1 / v for v in data))
    return res


def mean_trimmed(data: numeric, k: int) -> float:
    """return trimmed mean from data, k defines number of data to trim"""

    res = bar(sorted(data)[k:-k])
    return res


def median(data: numeric) -> float:
    """returns median number of data"""

    data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        i = int(n / 2)
        res = sum([data[i], data[i - 1]]) / 2
        return res
    else:
        i = int((n + 1) / 2) - 1
        return data[i]


def mode(data: numeric) -> list | None:
    """returns mode value from data"""

    cnt = {v: data.count(v) for v in set(data)}
    cntmax = max(cnt.values())
    if cntmax == 1:
        return None
    else:
        res = [v for v in cnt if cnt[v] == cntmax]
        return res


def data_range(data: numeric) -> float:
    """returns range of data"""

    res = max(data) - min(data)
    return res


def quantile(data: numeric, q: float) -> float:
    """returns quantile value from data"""

    data = sorted(data)
    k = (len(data) - 1) * q
    i = int(k)
    r = k - i
    res = data[i] * (1 - r) + data[i + 1] * r
    return res


def iqr_range(data: numeric) -> float:
    """returns IQR range from data"""

    res = quantile(data, 0.75) - quantile(data, 0.25)
    return res


def var(data: numeric, dof: int = 1) -> float:
    """returns variance of data"""

    b = bar(data)
    res = sum((d - b) ** 2 for d in data) / (len(data) - dof)
    return res


def std(data: numeric, dof: int = 1) -> float:
    """return standard deviation of data"""

    res = var(data, dof) ** (1/2)
    return res


def standardize(num: float, bar: float, std: float) -> float:
    """standardize value"""

    return (num - bar) / std


def scaler_standard(data: numeric, dof: int = 0) -> list:
    """returns standardized values of data"""

    b, s = bar(data), std(data, dof)
    res = [standardize(d, b, s) for d in data]
    return res


def variation(data: numeric, dof: int = 0) -> float:
    """returns coefficient of variation"""

    res = std(data, dof) / bar(data)
    return res


def skew(data: numeric, dof: int = 0) -> float:
    """returns skewness of data"""

    b, s = bar(data), std(data, dof)
    res = sum(standardize(d, b, s) ** 3 for d in data) / (len(data) - dof)
    return res


def skew_adv(data: numeric) -> float:
    """returns fixed skewness of data"""

    n = len(data)
    res = skew(data, 1) * (n / n - 2)
    return res


def kurtosis(data: numeric, dof: int = 0) -> float:
    """returns kurtosis of data"""

    b, s = bar(data), std(data, dof)
    res = (sum(standardize(d, b, s) ** 4 for d in data) / (len(data) - dof))
    return res


def kurtosis_norm(data: numeric, dof: int = 0) -> float:
    """returns kurtosis of normal distributed data"""

    res = kurtosis(data, dof) - 3
    return res


def kurtosis_adv(data: numeric) -> float:
    """returns fixed kurtosis of data"""

    k, n = kurtosis(data, 1), len(data)
    res = (k * n * (n + 1) / ((n - 2) * (n - 3))) - ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3)))
    return res


def jarque_bera(data: numeric, dof: int = 0) -> float:
    """returns Jarque-Bera normality test value"""

    res = (len(data) / 6) * (skew(data, dof) ** 2 + ((kurtosis_norm(data, dof) ** 2) / 4))
    return res


def cov(data_a: numeric, data_b: numeric, dof: int = 1) -> float:
    """returns covariance of two random variables"""

    b_a, b_b = bar(data_a), bar(data_b)
    res = sum((a - b_a) * (b - b_b) for a, b in zip(data_a, data_b)) / (len(data_a) - dof)
    return res


def pearson(data_a: numeric, data_b: numeric, dof: int = 0) -> float:
    """returns Pearson correlation coefficient of two data"""

    b_a, s_a = bar(data_a), std(data_a, dof)
    b_b, s_b = bar(data_b), std(data_b, dof)
    res = sum(standardize(a, b_a, s_a) * standardize(b, b_b, s_b) for a, b in zip(data_a, data_b)) / len(data_a) - dof
    return res


def corrcoef(a: numeric, b: numeric) -> float:
    """returns Pearson's r of two data"""

    res = cov(a, b) / ((cov(a, a) * cov(b, b)) ** 0.5)
    return res


def factorial(n: int) -> int:
    """returns factorial of number with for loop"""

    res = 1
    for i in range(1, n+1):
        res *= i
    return res


def factorial_rec(n: int) -> int:
    """returns factorial of number with recursion"""

    return 1 if n == 1 else n * factorial_rec(n - 1)


def permutation(n: int, k: int) -> float:
    """returns permutation of N things taken k at a time"""

    res = factorial(n) / factorial(n - k)
    return res


def combination(n: int, k: int) -> float:
    """returns combinations of N things taken k at a time"""

    res = factorial(n) / (factorial(n - k) * factorial(k))
    return res


def multiset(n: int, k: int) -> float:
    """return multiset of N things taken k at a time"""

    res = combination(n + k - 1, k)
    return res


def bernoulli_d(x: int, p: float, n: int = 1) -> float:
    """
    returns probability of bernoulli distribution
    x: case
    p: probability
    """

    res = (p ** x) * ((1 - p) ** (n - x))
    return res


def binom_d(x: int, n: int, p: float) -> float:
    """
    returns probability of binom distribution
    x: case
    n: number of trial
    p: probability
    """

    res = combination(n, x) * bernoulli_d(x=x, n=n, p=p)
    return res


def binom_c(x: int, n: int, p: float) -> float:
    """
    returns cumulative probability of binom distribution
    x: case
    n: number of trial
    p: probability
    """

    res = sum(binom_d(i, n, p) for i in range(x + 1))
    return res


def hyper_d(x: int, M: int, n: int, N: int) -> float:
    """
    returns probability of hypergeometric distribution
    x: case
    M: size of subpopulation
    n: size of sample
    N: size of population
    """

    res = combination(M, x) * combination(N - M, n - x) / combination(N, n)
    return res


def hyper_c(x: int, M: int, n: int, N: int) -> float:
    """
    returns cumulative probability of hypergeometric distribution
    x: case
    M: size of subpopulation
    n: size of sample
    N: size of population
    """

    res = sum(combination(M, x) * combination(N - M, n - x) / combination(N, n) for x in range(x + 1))
    return res


def pois_d(x: int, l: float) -> float:
    """
    returns probability of poisson distribution
    x: case
    l: lambda, expectation of random variable
    """

    res = (math.e ** -l) * (l ** x) / factorial(x)
    return res


def pois_c(x: int, l: float) -> float:
    """
    returns cumulative probability of poisson distribution
    x: case
    l: lambda, expectation of random variable
    """

    res = sum(pois_d(i, l) for i in range(x + 1))
    return res


def lineFit(x: numeric, y: numeric) -> tuple:
    """returns linear regression coefficient(weight) and intercept of two random variables"""

    x_bar, y_bar = bar(x), bar(y)
    tmp_0 = [(i - x_bar) * j for i, j in zip(x, y)]
    tmp_1 = [(i - x_bar) * i for i in x]
    w = sum(tmp_0) / sum(tmp_1)
    i = y_bar - (w * x_bar)
    return i, w


if __name__ == "__main__":
    a: numeric = [1, 2, 1, 3, 1, 2, 3, 1, 1, 2, 1, 1]
    print(f'\n{a=}\n')

    print(f'frequency of a: {freq(a)}')
    print(f'relative frequency of a: {freq_rel(a)}')
    print(f'cumulative frequency of a: {freq_cum(a)}')

    b: numeric = [2, 3, 4, 5, 6, 7, 1, 2, 3, 4]
    print(f'\n{b=}\n')

    print(f'deviation of each value of b: {devi(b)}')

    c: numeric = [1, 2, 3, 4, 5]
    d: numeric = [5, 4, 3, 2, 1]
    print(f'\n{c=}\n{d=}\n')

    print(f'weighted mean of c as data, d as weight: {mean_weight(c, d)}')
    print(f'product of c: {production(c)}')
    print(f'product of c: {production_rec(c)}')
    print(f'geometric mean of c: {mean_geom(c)}')
    print(f'harmonic mean of c: {mean_harm(c)}')
    print(f'trimmed mean of c, k=1: {mean_trimmed(c, 1)}')

    c: numeric = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    d: numeric = [1, 2, 3, 4, 5]
    print(f'\n{c=}\n{d=}\n')

    print(f'median of c: {median(c)}')
    print(f'median of d: {median(d)}')
    print(f'mode of c: {mode(c)}')
    print(f'mode of d: {mode(d)}')

    e: numeric = [-5, -2, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 17, 19, 25, 87, 99, 100]
    q: numeric = [0.25, 0.5, 0.75]
    print(f'\n{e=}\n{q=}\n')

    print(f'Q1, Q2, Q3 of e: {[quantile(e, q) for q in q]}')
    print(f'IQR range of e: {iqr_range(data=e)}')
    print(f'sample variance of e: {var(e, 1)}')
    print(f'variance of e: {var(e)}')
    print(f'standard deviation of e: {std(e, 1)}')
    print(f'coefficient of variation of e: {variation(e)}')
    print(f'skewness of e: {skew(e, 1)}')
    print(f'kurtosis of e: {kurtosis_norm(e, 1)}')

    f: numeric = [2.23, 4.78, 7.21, 9.37, 11.64, 14.23, 16.55, 18.70, 21.05, 23.21]
    g: numeric = [139, 123, 115, 96, 62, 54, 10, -3, -13, -55]
    print(f'\n{c=}\n{g=}\n')

    print(f'cov of a, b: {cov(f, g)}')
    print(f'correlation pearson: {pearson(f, g)}')
    print(f'correlation pearson: {corrcoef(f, g)}')

    print(f'factorial of 10: {factorial(10)}')
    print(f'factorial of 10: {factorial_rec(10)}')
    print(f'permutation of 10 things taken 7: {permutation(10, 7)}')
    print(f'combination of 10 things taken 7: {combination(10, 7)}')
    print(f'multiset of 10 things taken 7: {multiset(10, 7)}')

    print(f'probability of bernoulli distribution: {[bernoulli_d(i, 1/3) for i in range(4)]}')
    print(f'probability of binom distribution: {binom_d(8, 15, 0.5)}')
    print(f'cumulative probability of binom distribution: {binom_c(8, 15, 0.5)}')
    print(f'probability of hypergeometric distribution: {hyper_d(x=1, M=4, n=3, N=10)}')
    print(f'cumulative probability of hypergeometric distribution: {hyper_c(x=1, M=4, n=3, N=10)}')
    print(f'probability of poisson and binom distribution: {pois_d(x=2, l=2)}, {binom_d(x=2, n=20000, p=1/10000)}')
    print(f'cumulative probability of poisson and binom distribution: {pois_c(x=2, l=2)}, {binom_c(x=2, n=20000, p=1/10000)}')

    print(f'linear regression of a, b: {lineFit(f, g)}')