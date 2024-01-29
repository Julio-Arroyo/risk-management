import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import fredapi

# Market Yield on US Treasury Securities at different maturities
SERIES_NAMES = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3',
                'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
MATURITIES = [1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360]
DATE = "2023-12-29"

def get_yield_curve(date: str) -> List[float]:
    """
    Return a list. 'yield_curve[i]' corresponds to 'MATURITIES[i]'
    """
    fred = fredapi.Fred(api_key='fd97b1fdb076ff1a86aff9b38d7a0e70')
    yield_curve = []
    for series_name in SERIES_NAMES:
        yield_ = fred.get_series(series_name,
                                 observation_start=date,
                                 observation_end=date)
        yield_curve.append(yield_.values.squeeze())

    assert(len(yield_curve) == len(MATURITIES))
    return yield_curve

def interpolate_yields(obs_yield_curve: List[float]) -> List[float]:
    """
    Use straight-line interpolation between observed yields.
    """
    interp_yield_curve = []
    for i in range(len(obs_yield_curve) - 1):
        T_curr, T_next = MATURITIES[i], MATURITIES[i+1]
        yield_curr, yield_next = (obs_yield_curve[i],
                                  obs_yield_curve[i+1])
        m = (yield_next - yield_curr)/(T_next - T_curr)

        interp_yield_curve.append(yield_curr)
        for T in range(T_curr+1, T_next):
            interp_yield = m*(T-T_curr) + yield_curr
            interp_yield_curve.append(interp_yield)
    interp_yield_curve.append(obs_yield_curve[-1])  # last obs yield

    assert(len(interp_yield_curve) == 360)
    return interp_yield_curve

def fit_Nelson_Siegel(interp_yield_curve: List[float],
                      objective="absolute-differences",
                      method="linear-programming") \
                      -> Tuple[float, float, float, float]:
    """
    Following Nelson-Siegel 1987 paper.

    Define a grid of tao's. Then, the problem becomes a linear regression
    to find betas.

    Return:
        - Nelson-Siegel parameters: (b0, b1, b2, tao)
    """

    def transform_maturities(tao: float) -> np.ndarray:
        """
        Simplify (Eq 2.) in Nelson-Siegel such that there is only one
        of each beta. The resulting coefficients of the betas are the
        values stored in X.
        """
        N = 360
        X = np.zeros((N, 3))
        for T in range(1, N+1):
            frac = T/tao
            X[T-1, 0] = 1
            X[T-1, 1] = (1/frac)*(1 - np.exp(-1*frac))
            X[T-1, 2] = (1/frac)*(1 - (1+frac)*np.exp(-1*frac))
        return X


    def min_abs_diff_lp(X: np.ndarray,
                        Y: np.ndarray) \
                        -> (Tuple[float, float, float], float):
        import cvxpy as cp

        Betas = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.norm(Y - X@Betas, 1)))

        min_abs_diff = prob.solve()
        betas_soln = Betas.value[0], Betas.value[1], Betas.value[2]

        return betas_soln, min_abs_diff

    if (objective == "absolute-differences" and
        method == "linear-programming"):
        print("Minimizing sum of absolute differences using linear programming...\n")
        solver = min_abs_diff_lp
    else:
        raise NotImplementedError(f"Objective '{objective}' has \
                                    not been implemented")

    Y = np.array(interp_yield_curve)
    # grid used by Nelson-Siegel (1987)
    tao_grid = list(range(10, 201, 10)) + [250, 300, 365]

    min_loss = float('inf')
    best_betas, best_tao = None, None

    for tao in tao_grid:
        X = transform_maturities(tao)
        betas, loss = solver(X, Y)
        if loss < min_loss:
            min_loss = loss
            best_tao = tao
            best_betas = betas

    return (best_betas[0], best_betas[1], best_betas[2], best_tao)

if __name__ == "__main__":
    obs_yield_curve = get_yield_curve(DATE)
    interp_yield_curve = interpolate_yields(obs_yield_curve)
    (b0, b1, b2, tao) = fit_Nelson_Siegel(interp_yield_curve)
    print(f"Nelson-Siegel parameters")
    print(f"\t-Beta_0: {b0}")
    print(f"\t-Beta_1: {b1}")
    print(f"\t-Beta_2: {b2}")
    print(f"\t-Tao: {tao}")

    plt.scatter(MATURITIES, obs_yield_curve,
                marker='D', color='black', label='Observed')
    plt.scatter(range(1, 360+1, 10), interp_yield_curve[::10],
                marker='X', color='black', s=20, linewidths=1,
                label='Interpolated')

    def r(f):
        return (b0
                + b1*(tao/f)*(1 - np.exp(-f/tao))
                + b2*(tao/f)*(1 - np.exp(-f/tao)*(1+f/tao)))
    f = np.linspace(1, 360, 100)
    plt.plot(f, r(f), color='red', label='Nelson-Siegel')

    plt.title ("US Treasury Yield Curve " + DATE)
    plt.xlabel("Maturity (months)")
    plt.ylabel("Market Yield (%)")
    plt.legend()
    plt.show()

