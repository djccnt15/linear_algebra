def freq(data: list) -> dict:
    """returns frequency of each value"""

    res = {val: data.count(val) for val in set(data)}
    return res


def freq_rel(data: list) -> dict:
    """returns relative frequency of each value"""

    res = {val: data.count(val) / len(data) for val in set(data)}
    return res


def freq_cum(data: list) -> dict:
    """returns cumulative frequency of sorted value"""

    data_sort = sorted(list(set(data)))
    res = {val: sum(data.count(j) for j in data_sort[:i+1]) for i, val in enumerate(data_sort)}
    return res


def bar(data: list) -> float:
    """returns expectation/sample mean"""

    res = sum(data) / len(data)
    return res


def devi(data: list) -> dict:
    """returns deviation of each value"""

    b = bar(data)
    res = {val: val - b for val in sorted(list(set(data)))}
    return res


def mean_weight(data: list, weights: list) -> float:
    """returns weighted mean"""

    res = sum(v * w for v, w in zip(data, weights)) / sum(weights)
    return res


def prod_for(data: list) -> float:
    """product all elements in data with for loop"""

    res = 1
    for i in data:
        res *= i
    return res


def prod_rec(data: list) -> float:
    """product all elements in data with recursion"""

    return data[0] if len(data) == 1 else data[0] * prod_rec(data[1:])


def mean_geom(data: list) -> float:
    """returns geometric mean of data"""

    res = prod_rec(data) ** (1 / len(data))
    return res


def mean_harm(data: list) -> float:
    """returns harmonic mean of data"""

    res = len(data) / (sum(1 / v for v in data))
    return res


def mean_trimmed(data: list, k: int) -> float:
    """return trimmed mean from data, k defines number of data to trim"""

    res = bar(sorted(data)[k:-k])
    return res


def median(data: list) -> float:
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


def mode(data: list) -> list | None:
    """returns mode value from data"""

    cnt = {v: data.count(v) for v in set(data)}
    cntmax = max(cnt.values())
    if cntmax == 1:
        return None
    else:
        res = [v for v in cnt if cnt[v] == cntmax]
        return res


def data_range(data: list) -> float:
    """returns range of data"""

    res = max(data) - min(data)
    return res


def quantile(data: list, q: float) -> float:
    """returns quantile value from data"""

    data = sorted(data)
    k = (len(data) - 1) * q
    i = int(k)
    r = k - i
    res = data[i] * (1 - r) + data[i + 1] * r
    return res


def iqr_range(data: list) -> float:
    """returns IQR range from data"""

    res = quantile(data, 0.75) - quantile(data, 0.25)
    return res


def var(data: list, dof: int = 0) -> float:
    """returns variance of data"""

    b = bar(data)
    res = sum((d - b) ** 2 for d in data) / (len(data) - dof)
    return res


def std(data: list, dof: int = 0) -> float:
    """return standard deviation of data"""

    res = var(data, dof) ** (1/2)
    return res


def standardize(num: float, bar: float, std: float) -> float:
    """standardize value"""

    return (num - bar) / std


def scaler_standard(data: list) -> list:
    """returns standardized values of data"""

    b, s = bar(data), std(data)
    res = [standardize(d, b, s) for d in data]
    return res


def variation(data: list) -> float:
    """returns coefficient of variation"""

    res = std(data) / bar(data)
    return res


def skew(data: list, dof: int = 0) -> float:
    """returns skewness of data"""

    b, s = bar(data), std(data)
    res = sum(standardize(d, b, s) ** 3 for d in data) / (len(data) - dof)
    return res


def skew_adv(data: list) -> float:
    """returns fixed skewness of data"""

    n = len(data)
    res = skew(data, 1) * (n / n - 2)
    return res


def kurtosis(data: list, dof: int = 0) -> float:
    """returns kurtosis of data"""

    b, s = bar(data), std(data)
    res = (sum(standardize(d, b, s) ** 4 for d in data) / (len(data) - dof))
    return res


def kurtosis_norm(data: list, dof: int = 0) -> float:
    """returns kurtosis of normal distributed data"""

    res = kurtosis(data, dof) - 3
    return res


def kurtosis_adv(data: list) -> float:
    """returns fixed kurtosis of data"""

    k, n = kurtosis(data, 1), len(data)
    res = (k * n * (n + 1) / ((n - 2) * (n - 3))) - ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3)))
    return res


def jarque_bera(data: list, dof: int = 0) -> float:
    """returns Jarque-Bera normailty test value"""

    res = (len(data) / 6) * (skew(data, dof) ** 2 + ((kurtosis_norm(data, dof) ** 2) / 4))
    return res


def cov(data_a: list, data_b: list, dof: int = 0) -> float:
    """returns covariance of two random variables"""

    b_a, b_b = bar(data_a), bar(data_b)
    res = sum((a - b_a) * (b - b_b) for a, b in zip(data_a, data_b)) / (len(data_a) - dof)
    return res


def pearson(data_a: list, data_b: list, dof: int = 0) -> float:
    """returns Pearson correlation coefficient of two data"""

    b_a, s_a = bar(data_a), std(data_a)
    b_b, s_b = bar(data_b), std(data_b)
    res = sum(standardize(a, b_a, s_a) * standardize(b, b_b, s_b) for a, b in zip(data_a, data_b)) / len(data_a) - dof
    return res


def corrcoef(a: list, b: list) -> float:
    """returns Pearson's r of two data"""

    res = cov(a, b) / ((cov(a, a) * cov(b, b)) ** 0.5)
    return res


def lineFit(x: list, y: list) -> tuple:
    """returns linear regression coefficient(weight) and intercept of two random variables"""

    x_bar, y_bar = bar(x), bar(y)
    tmp_0 = [(i - x_bar) * j for i, j in zip(x, y)]
    tmp_1 = [(i - x_bar) * i for i in x]
    w = sum(tmp_0) / sum(tmp_1)
    i = y_bar - (w * x_bar)
    return i, w


if __name__ == "__main__":
    a = [1, 2, 1, 3, 1, 2, 3, 1, 1, 2, 1, 1]
    print(f'\n{a=}\n')

    print(f'frequency of a: {freq(a)}')
    print(f'relative frequency of a: {freq_rel(a)}')
    print(f'cumulative frequency of a: {freq_cum(a)}')

    b = [2, 3, 4, 5, 6, 7, 1, 2, 3, 4]
    print(f'\n{b=}\n')

    print(f'deviation of each value of b: {devi(b)}')

    c = [1, 2, 3, 4, 5]
    d = [5, 4, 3, 2, 1]
    print(f'\n{c=}\n{d=}\n')

    print(f'weighted mean of c as data, d as weight: {mean_weight(c, d)}')
    print(f'prod of c: {prod_rec(c)}')
    print(f'geometric mean of c: {mean_geom(c)}')
    print(f'harmonic mean of c: {mean_harm(c)}')
    print(f'trimmed mean of c, k=1: {mean_trimmed(c, 1)}')

    c = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    d = [1, 2, 3, 4, 5]
    print(f'\n{c=}\n{d=}\n')

    print(f'median of c: {median(c)}')
    print(f'median of d: {median(d)}')
    print(f'mode of c: {mode(c)}')
    print(f'mode of d: {mode(d)}')

    e = [-5, -2, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 17, 19, 25, 87, 99, 100]
    q = [0.25, 0.5, 0.75]
    print(f'\n{e=}\n{q=}\n')

    print(f'Q1, Q2, Q3 of e: {[quantile(e, q) for q in q]}')
    print(f'IQR range of e: {iqr_range(data=e)}')
    print(f'sample variance of e: {var(e, 1)}')
    print(f'variance of e: {var(e)}')
    print(f'standard deviation of e: {std(e, 1)}')
    print(f'coefficient of variation of e: {variation(e)}')
    print(f'skewness of e: {skew(e, 1)}')
    print(f'kurtosis of e: {kurtosis_norm(e, 1)}')

    f = [2.23, 4.78, 7.21, 9.37, 11.64, 14.23, 16.55, 18.70, 21.05, 23.21]
    g = [139, 123, 115, 96, 62, 54, 10, -3, -13, -55]
    print(f'\n{c=}\n{g=}\n')

    print(f'cov of a, b: {cov(f, g)}')
    print(f'correlation pearson: {pearson(f, g)}')
    print(f'correlation pearson: {corrcoef(f, g)}')

    print(f'linear regression of a, b: {lineFit(f, g)}')