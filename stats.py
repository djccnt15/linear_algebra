import math

from common import *

numeric = vector


def freq(data: numeric) -> dict:
    """returns frequency of each value"""

    return {val: data.count(val) for val in set(data)}


def freq_rel(data: numeric) -> dict:
    """returns relative frequency of each value"""

    return {val: data.count(val) / len(data) for val in set(data)}


def freq_cum(data: numeric) -> dict:
    """returns cumulative frequency of sorted value"""

    data_sort = sorted(list(set(data)))
    return {val: sum(data.count(j) for j in data_sort[:i + 1]) for i, val in enumerate(data_sort)}


def bar(data: numeric) -> float:
    """returns expectation/sample mean"""

    return sum(data) / len(data)


def devi(data: numeric) -> dict:
    """returns deviation of each value"""

    return {val: val - bar(data) for val in sorted(list(set(data)))}


def mean_weight(data: numeric, weights: numeric) -> float:
    """returns weighted mean"""

    return sum(v * w for v, w in zip(data, weights)) / sum(weights)


def mean_geom(data: numeric) -> float:
    """returns geometric mean of data"""

    return production(data) ** (1 / len(data))


def mean_harm(data: numeric) -> float:
    """returns harmonic mean of data"""

    return len(data) / (sum(1 / v for v in data))


def mean_trimmed(data: numeric, k: int) -> float:
    """return trimmed mean from data, k defines number of data to trim"""

    return bar(sorted(data)[k:-k])


def median(data: numeric) -> float:
    """returns median number of data"""

    data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        i = int(n / 2)
        return sum([data[i], data[i - 1]]) / 2
    else:
        return data[int((n + 1) / 2) - 1]


def mode(data: numeric) -> list | None:
    """returns mode value from data"""

    cnt = {v: data.count(v) for v in set(data)}
    cntmax = max(cnt.values())
    if cntmax == 1:
        return None
    else:
        return [v for v in cnt if cnt[v] == cntmax]


def data_range(data: numeric) -> float:
    """returns range of data"""

    return max(data) - min(data)


def quantile(data: numeric, q: float) -> float:
    """returns quantile value from data"""

    data = sorted(data)
    k = (len(data) - 1) * q
    i = int(k)
    r = k - i
    return data[i] * (1 - r) + data[i + 1] * r


def iqr_range(data: numeric) -> float:
    """returns IQR range from data"""

    return quantile(data, 0.75) - quantile(data, 0.25)


def var(data: numeric, dof: int = 1) -> float:
    """returns variance of data"""

    return sum((d - bar(data)) ** 2 for d in data) / (len(data) - dof)


def std(data: numeric, dof: int = 1) -> float:
    """return standard deviation of data"""

    return var(data, dof) ** (1 / 2)


def standardize(num: float, bar: float, std: float) -> float:
    """standardize value"""

    return (num - bar) / std


def scaler_standard(data: numeric, dof: int = 0) -> list:
    """returns standardized values of data"""

    b, s = bar(data), std(data, dof)
    return [standardize(d, b, s) for d in data]


def variation(data: numeric, dof: int = 0) -> float:
    """returns coefficient of variation"""

    return std(data, dof) / bar(data)


def skew(data: numeric, dof: int = 0) -> float:
    """returns skewness of data"""

    b, s = bar(data), std(data, dof)
    return sum(standardize(d, b, s) ** 3 for d in data) / (len(data) - dof)


def skew_adv(data: numeric) -> float:
    """returns fixed skewness of data"""

    n = len(data)
    return skew(data, 1) * (n / n - 2)


def kurtosis(data: numeric, dof: int = 0) -> float:
    """returns kurtosis of data"""

    b, s = bar(data), std(data, dof)
    return (sum(standardize(d, b, s) ** 4 for d in data) / (len(data) - dof))


def kurtosis_norm(data: numeric, dof: int = 0) -> float:
    """returns kurtosis of normal distributed data"""

    return kurtosis(data, dof) - 3


def kurtosis_adv(data: numeric) -> float:
    """returns fixed kurtosis of data"""

    k, n = kurtosis(data, 1), len(data)
    return (k * n * (n + 1) / ((n - 2) * (n - 3))) - ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3)))


def jarque_bera(data: numeric, dof: int = 0) -> float:
    """returns Jarque-Bera normality test value"""

    return (len(data) / 6) * (skew(data, dof) ** 2 + ((kurtosis_norm(data, dof) ** 2) / 4))


def cov(data_a: numeric, data_b: numeric, dof: int = 1) -> float:
    """returns covariance of two random variables"""

    b_a, b_b = bar(data_a), bar(data_b)
    return sum((a - b_a) * (b - b_b) for a, b in zip(data_a, data_b)) / (len(data_a) - dof)


def pearson(data_a: numeric, data_b: numeric, dof: int = 0) -> float:
    """returns Pearson correlation coefficient of two data"""

    b_a, s_a = bar(data_a), std(data_a, dof)
    b_b, s_b = bar(data_b), std(data_b, dof)
    return sum(standardize(a, b_a, s_a) * standardize(b, b_b, s_b) for a, b in zip(data_a, data_b)) / len(data_a) - dof


def corrcoef(a: numeric, b: numeric) -> float:
    """returns Pearson's r of two data"""

    return cov(a, b) / ((cov(a, a) * cov(b, b)) ** 0.5)


def permutation(n: int, k: int) -> float:
    """returns permutation of N things taken k at a time"""

    return factorial(n) / factorial(n - k)


def combination(n: int, k: int) -> float:
    """returns combinations of N things taken k at a time"""

    return factorial(n) / (factorial(n - k) * factorial(k))


def multiset(n: int, k: int) -> float:
    """return multiset of N things taken k at a time"""

    return combination(n + k - 1, k)


def bernoulli_d(p: float, x: int = 0 | 1, n: int = 1) -> float:
    """
    returns probability of bernoulli distribution
    x: case
    p: probability
    """

    return (p ** x) * ((1 - p) ** (n - x))


def binom_d(x: int, n: int, p: float) -> float:
    """
    returns probability of binom distribution
    x: case
    n: number of trial
    p: probability
    """

    return combination(n, x) * bernoulli_d(x=x, n=n, p=p)


def binom_c(x: int, n: int, p: float, start: int = 0) -> float:
    """
    returns cumulative probability of binom distribution
    x: case
    n: number of trial
    p: probability
    """

    return sum(binom_d(i, n, p) for i in range(start, x + 1))


def hyper_d(x: int, M: int, n: int, N: int) -> float:
    """
    returns probability of hypergeometric distribution
    x: case
    M: size of subpopulation
    n: size of sample
    N: size of population
    """

    return combination(M, x) * combination(N - M, n - x) / combination(N, n)


def hyper_c(x: int, M: int, n: int, N: int, start: int = 0) -> float:
    """
    returns cumulative probability of hypergeometric distribution
    x: case
    M: size of subpopulation
    n: size of sample
    N: size of population
    """

    return sum(hyper_d(x=x, n=n, N=N, M=M) for x in range(start, x + 1))


def pois_d(x: int, l: float) -> float:
    """
    returns probability of poisson distribution
    x: case
    l: lambda, expectation of random variable
    """

    return (math.e ** -l) * (l ** x) / factorial(x)


def pois_c(x: int, l: float, start: int = 0) -> float:
    """
    returns cumulative probability of poisson distribution
    x: case
    l: lambda, expectation of random variable
    """

    return sum(pois_d(i, l) for i in range(start, x + 1))


def geom_d(x: int, p: float) -> float:
    """
    returns probability of geometric distribution
    x: number of failures
    p: probability
    """

    return ((1 - p) ** x) * p


def geom_c(x: int, p: float) -> float:
    """
    returns cumulative probability of geometric distribution
    x: number of failures
    p: probability
    """

    return 1 - ((1 - p) ** (x + 1))


def nbinom_d(x: int, r: int, p: float) -> float:
    """
    returns probability of negative binomial distribution
    x: number of failures
    r: number of success
    p: probability
    """

    return combination(x + r - 1, r - 1) * (p ** r) * ((1 - p) ** x)


def nbinom_c(x: int, r: int, p: float, start: int = 0) -> float:
    """
    returns cumulative probability of negative binomial distribution
    x: number of failures
    r: number of success
    p: probability
    """

    return sum(nbinom_d(i, r, p) for i in range(start, x + 1))


def multi_d(x: list[int], p: list[float]) -> float:
    """
    returns probability of multinomial distribution
    x: cases
    p: probability of each case
    """

    return factorial(sum(x)) / production(factorial(v) for v in x) * production(p ** x for p, x in zip(p, x))  # type: ignore


def lineFit(x: numeric, y: numeric) -> tuple:
    """returns linear regression coefficient(weight) and intercept of two random variables"""

    x_bar, y_bar = bar(x), bar(y)
    w = sum((x - x_bar) * y for x, y in zip(x, y)) / sum((x - x_bar) * x for x in x)
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

    print(f'probability of bernoulli distribution: {[bernoulli_d(x=i, p=0.2) for i in range(2)]}')
    print(f'probability of binom distribution: {binom_d(8, 15, 0.5)}')
    print(f'cumulative probability of binom distribution: {binom_c(8, 15, 0.5)}')
    print(f'probability of hypergeometric distribution: {hyper_d(x=1, M=4, n=3, N=10)}')
    print(f'cumulative probability of hypergeometric distribution: {hyper_c(x=1, M=4, n=3, N=10)}')
    print(f'probability of poisson and binom distribution: {pois_d(x=2, l=2)}, {binom_d(x=2, n=20000, p=1/10000)}')
    print(f'cumulative probability of poisson and binom distribution: {pois_c(x=2, l=2)}, {binom_c(x=2, n=20000, p=1/10000)}')
    print(f'probability of geometric distribution: {geom_d(x=3, p=0.3)}')
    print(f'cumulative probability of geometric distribution: {geom_c(x=5, p=0.3)}')
    print(f'probability of negative binomial distribution: {nbinom_d(x=4, r=3, p=0.3)}')
    print(f'cumulative probability of negative binomial distribution: {nbinom_d(x=4, r=3, p=0.3)}')
    print(f'probability of multinomial distribution: {multi_d(x=[5, 6, 9], p=[0.3, 0.4, 0.3])}')

    print(f'linear regression of a, b: {lineFit(f, g)}')