"""
Author: Julio Arroyo 2024

Given spreads over US Treasury curve of A-rated IBM corporates, find
best-fitting "'quasi'-debt to firm value ratio" ('d') and enterprise
volatility ('\sigma_V'). Assume Merton Spread Model holds.
"""


import matplotlib.pyplot as plt
import scipy.stats as spst
import scipy.optimize
import numpy as np


def merton_spread(T: np.ndarray, sigma_V: float, d: float):
    s = sigma_V * np.sqrt(T)

    d1 = (-np.log(d) + np.power(s, 2)/2)/s
    d2 = d1 - s

    log_arg = spst.norm.cdf(-d1)/d + spst.norm.cdf(d2)
    return -np.log(log_arg)/T


if __name__ == "__main__":
    # July 1, 2021 spreads over US Treasury Curve for A-rated IBM
    spreads = [-6.3, 14.5, 26.9, 33.5, 49.5, 82.7, 83.1]  # bps/year
    T = [1, 3, 5, 7, 10, 20, 30]  # Years

    spreads = (1/100)*np.array(spreads)
    opt_params, _ = scipy.optimize.curve_fit(merton_spread, T, spreads)
    print(opt_params)

    plt.scatter(T, spreads, label='data')
    plt.plot(T, merton_spread(T, opt_params[0], opt_params[1]), label='fit')
    plt.legend()
    plt.show()

