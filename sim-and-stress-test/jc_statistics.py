from typing import Tuple
import scipy.stats
import numpy as np


def central_moment(x: np.ndarray, i: int) -> float:
    """
    Compute the i-th central moment of N samples

    $ m_i = \sum_{i=1}^N (x_i - x_bar)^i $
    """
    assert(len(x.shape) == 1)  # one-dimensional

    N = x.shape[0]
    x_bar = np.mean(x)

    x_centered = x - x_bar
    m_i = np.mean(np.power(x_centered, i))

    # m_i = 0
    # for i in range(N):
    #     m_i += np.power((x[i] - x_bar), i)
    # m_i /= N

    return m_i


def skewness(x: np.ndarray) -> float:
    """
    Sample skewness is computed as the Fisher-Pearson coefficient of skewness:

    skew = m3/sqrt(m2)
    """
    m3 = central_moment(x, 3)
    m2 = central_moment(x, 2)
    np.testing.assert_allclose([m2, m3],
                               [scipy.stats.moment(x, 2), scipy.stats.moment(x, 3)])
    # assert(m3 == scipy.stats.moment(x, 3))
    # assert(m2 == scipy.stats.moment(x, 2))

    skew = m3/np.sqrt(np.power(m2, 3))
    np.testing.assert_allclose(skew, scipy.stats.skew(x))

    return skew


def kurtosis(x: np.ndarray, excess=True) -> float:
    """
    Compute the sample kurtosis
    """
    m4 = central_moment(x, 4)
    m2 = central_moment(x, 2)

    kurt = m4/np.power(m2, 2)
    if excess:
        kurt -= 3

    np.testing.assert_allclose(kurt, scipy.stats.kurtosis(x))

    return kurt

def jarque_bera_test(x: np.ndarray) -> Tuple[float, float]:
    assert(len(x.shape) == 1)

    n = x.shape[0]
    skew = skewness(x)
    excess_kurtosis = kurtosis(x)

    s2 = np.power(skew, 2)
    k2 = np.power(excess_kurtosis, 2)

    jb = (n/6)*(s2 + k2/4)
    pr_normality = 1 - scipy.stats.chi2.cdf(jb, df=2)

    statistic, pval = scipy.stats.jarque_bera(x)
    np.testing.assert_allclose([jb, pr_normality],
                               [statistic, pval])

    return (jb, pr_normality)

