import matplotlib.pyplot as plt
from typing import Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np
import fredapi


# Market Yield on US Treasury Securities at different maturities
SERIES_NAMES = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3',
                'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
START_DATE = "2022-12-30"
END_DATE = "2023-12-29"


def get_yield_data() -> Tuple[List[str], np.ndarray]:
    """
    Return:
        - dates: at which yields are observed
        - yields_mat: columns dates, rows maturities
    """
    fred = fredapi.Fred(api_key='fd97b1fdb076ff1a86aff9b38d7a0e70')
    first = True
    for series_name in SERIES_NAMES:
        yield_series = fred.get_series(series_name,
                                       observation_start=START_DATE,
                                       observation_end=END_DATE)
        yield_series.rename(series_name)
        if first:
            df = pd.DataFrame(yield_series)
            first = False
        else:
            df = pd.concat([df, yield_series], axis=1)

    cdates_dirty = df.index.strftime('%Y-%m-%d').tolist()
    yields_mat_dirty = [list(df.iloc[i]) for i in range(len(df))]

    # clean up
    is_nan = [any(np.isnan(yields)) for yields in yields_mat_dirty]
    dates = [cdates_dirty[i] for i in range(len(cdates_dirty)) if not is_nan[i]]
    yields_mat = np.array([yields_mat_dirty[i] for i in range(len(yields_mat_dirty)) if not is_nan[i]])

    return(dates, yields_mat)

def calc_deltas(dates: List[str],
                yield_mat: np.ndarray) -> np.ndarray:
    "Return matrix with deltas in yields, day-over-day"
    assert(len(dates) == len(yield_mat))

    delta_yields = np.zeros((yield_mat.shape[0]-1,
                             yield_mat.shape[1]))
    for i in range(len(dates) - 1):
        date1 = datetime.strptime(dates[i], '%Y-%m-%d')
        date2 = datetime.strptime(dates[i+1], '%Y-%m-%d')
        n_days = (date2 - date1).days
        delta_yields[i,:] = (1/n_days)*(yield_mat[i+1,:] - yield_mat[i,:])
    return delta_yields

def calc_cov(delta_yields: np.ndarray) -> np.ndarray:
    def calc_sample_cov(u: np.ndarray, v: np.ndarray):
        assert(u.shape == v.shape)
        N = u.shape[0]
        assert(N > 1)

        u_bar, v_bar = np.mean(u), np.mean(v)
        cov_uv = 0
        for i in range(N):
            cov_uv += (u[i] - u_bar)*(v[i] - v_bar)
        return cov_uv/(N-1)

    N, T = delta_yields.shape
    cov = np.zeros((T, T))
    for i in range(0, T):
        for j in range(i, T):
            cov_ij = calc_sample_cov(delta_yields[:, i], delta_yields[:, j])
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij
    return cov


if __name__ == "__main__":
    dates, rate_mat = get_yield_data()
    delta_yields = calc_deltas(dates, rate_mat)
    cov_mat = calc_cov(delta_yields)

    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print(f"Percentage of trace from first three \
            Principal Components: {100*np.sum(eigen_vals[:3]/np.trace(cov_mat)):2f}%")

    plt.plot(list(range(1, 12)), eigen_vecs[:, 0], label='PC1')
    plt.plot(list(range(1, 12)), eigen_vecs[:, 1], label='PC2')
    plt.plot(list(range(1, 12)), eigen_vecs[:, 2], label='PC3')
    plt.ylabel("Level")
    plt.xlabel("Tenor")
    plt.xticks(list(range(1, 12)), ['1mo', '3mo', '6mo', '1yr', '2yr', '3yr', '5yr',
                '7yr', '10yr', '20yr', '30yr'])
    plt.title("Principal Components: Covariance of day-over-day yield deltas 12/2022-12/2023")
    plt.legend()
    plt.show()

