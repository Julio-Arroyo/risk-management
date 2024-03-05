import scipy.optimize
import numpy as np


q = 0.02  # probability of default
T = 30    # maturity in years
R = 50    # Recovery given default
COUPON = 2
PRINCIPAL = 100
SOFR = (1/100)*np.array([6 - 5*np.exp(-t/30) for t in range(1, T+1)])
P_def = None


def get_price_defaulting(q: float, r: np.ndarray):
    val = 0
    for i in range(1, T+1):
        val += np.power(1 - q, i)/np.power(1 + r[i-1], i)

    val *= (q*R/(1-q) + COUPON)
    val += PRINCIPAL*np.power(1-q, T)/np.power(1+r[T-1], T)
    return val


def equation(P_def: float):
    return P_def - get_price_defaulting(q, SOFR)


def get_price2(r: np.ndarray, s: float) -> float:
    """Calculate P(r, s)"""
    assert(np.all(r < 1))
    # assert(s < 1)

    price = 0
    for t in range(1, T + 1):
        price += (100*r[t-1] + 3)*np.exp(-(r[t-1] + s) * t)

    price += PRINCIPAL*np.exp(-(r[T-1] + s) * T)

    return price


def calc_error(s: float):
    return P_def - get_price2(SOFR, s)


if __name__ == "__main__":
    P_def = scipy.optimize.fsolve(equation, 0.5)[0]
    print(P_def)

    OAS = scipy.optimize.fsolve(calc_error,
                                (0.01))[0]  # initial guess
    print(OAS)


