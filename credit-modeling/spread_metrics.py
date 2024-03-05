import scipy.optimize
import numpy as np


T = 30  # maturity in years

SOFR = (1/100)*np.array([6 - 5*np.exp(-t/30) for t in range(1, T+1)])
COUPONS = (100*SOFR + 3)
PRINCIPAL = 100
PRICE = 87.62143198740198


# def get_price(r: np.ndarray, s: float) -> float:
#     """Calculate P(r, s)"""
#     assert(np.all(r < 1))
#     # assert(s < 1)
# 
#     price = 0
#     for t in range(1, T + 1):
#         price += COUPONS[t-1]*np.exp(-(r[t-1] + s) * t)
# 
#     price += PRINCIPAL*np.exp(-(r[T-1] + s) * T)
# 
#     return price


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
    return PRICE - get_price2(SOFR, s)

if __name__ == "__main__":
    # calculate root of error equation
    OAS = scipy.optimize.fsolve(calc_error,
                                (0.01))  # initial guess
    OAS = OAS[0]
    print(f"PART A)\nANSWER: the option-adjusted spread is: {OAS*100}%\n")
    print(f"Sanity Check: Plug OAS into price formula (should be equal to 87.62143198740198)")
    print(f"\t Res: {get_price2(SOFR, OAS)}")

    print(f"\nPart B)")
    perturbation = 0.0001

    # calculate Effective Duration
    numerator = get_price2(SOFR - perturbation, OAS) - get_price2(SOFR + perturbation, OAS)
    denominator = 2*perturbation*get_price2(SOFR, OAS)
    print(f"Effective Duration is {numerator/denominator}")

    # calculate OASD
    numerator = get_price2(SOFR, OAS - perturbation) - get_price2(SOFR, OAS + perturbation)
    denominator = 2*perturbation*get_price2(SOFR, OAS)
    print(f"Spread Duration is {numerator/denominator}")

