import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fredapi

SERIES_NAMES = ['DEXSZUS',  # Swiss Francs to One US dollar
                'DEXUSUK',  # US dollars to One UK Pound Sterling
                'DEXJPUS']  # Japanese Yen to One US dollar
LAST_YEAR_END = '2023-12-29'

def get_currency_data() -> pd.DataFrame:
    """
    Return a DataFrame of USD/currency

    Convert those exchange rates given in currency/USD
    Drop those dates where data is NaN
    """
    fred = fredapi.Fred(api_key='')

    inverted = [True, False, True]  # signals whether series is given in currency/USD
    labels = ['USD/Swissies', 'USD/Pounds', 'USD/Yen']

    df_currencies = pd.DataFrame()
    for i in range(len(SERIES_NAMES)):
        currency_series = fred.get_series(SERIES_NAMES[i],
                                          observation_start=None,
                                          observation_end=LAST_YEAR_END)
        # make sure dataframe has data in USD/currency
        if inverted[i]:
            df_currencies[labels[i]] = currency_series.apply(lambda x: 1/x)
        else:
            assert(not inverted[i])
            df_currencies[labels[i]] = currency_series

    return df_currencies.dropna()  # drop dates with at least one NaN

def to_log(df_currencies: pd.DataFrame) -> pd.DataFrame:
    df_log_currencies = pd.DataFrame()
    for exchange_rate_name in df_currencies.columns.to_list():
        log_er_name = f"LOG({exchange_rate_name})"

        today_rate = df_currencies[exchange_rate_name].iloc[0:-1]
        tomorrow_rate = df_currencies[exchange_rate_name].iloc[1:]
        log_returns = np.log(tomorrow_rate.values/today_rate.values)

        df_log_currencies[log_er_name] = pd.Series(log_returns,
                                                   index=tomorrow_rate.index.tolist())
    print(df_log_currencies)
    return df_log_currencies

def learning_holdout_split(df: pd.DataFrame) -> pd.DataFrame:
    """Keep last year as holdout set"""
    holdout_year = int(LAST_YEAR_END[:4])
    split_point = None
    for i in range(len(df.index)):
        date = str(df.index[i])  # in YYYY-MM-DD format
        year = int(date[:4])
        if year == holdout_year:
            split_point = i
            break
    return df.iloc[0:split_point], df.iloc[split_point:]

def get_expected_returns(log_returns: pd.DataFrame) -> np.ndarray:
    """Naively compute sample mean, use it as expected values"""
    m = np.mean(log_returns.values, axis=0)
    assert(m.shape == (3,))
    return m

def get_cov_mat(log_returns: pd.DataFrame) -> np.ndarray:
    C = np.cov(np.transpose(log_returns.values))
    assert(C.shape == (3, 3))
    return C

def get_efficient_portfolio(lambda1: float, m: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Eq. 4 QRPM"""
    n = C.shape[0]

    I = np.identity(n)
    J = np.ones((n, n))
    u = np.ones((n,))
    C_inv = np.linalg.inv(C)

    ut_Cinv_u = np.matmul(np.transpose(u), np.matmul(C_inv, u))
    Cinv_J = np.matmul(C_inv, J)
    Cinv_u = np.matmul(C_inv, u)
    Cinv_m = np.matmul(C_inv, m)

    return lambda1*np.matmul((I - (1/ut_Cinv_u)*(Cinv_J)), Cinv_m) + (1/ut_Cinv_u)*(Cinv_u)

def get_min_variance_portfolio(C: np.ndarray) -> np.ndarray:
    C_inv = np.linalg.inv(C)
    u = np.ones((C.shape[0]))

    Cinv_u = np.matmul(C_inv, u)
    ut_Cinv_u = np.matmul(np.transpose(u), Cinv_u)

    return (1/ut_Cinv_u)*Cinv_u

def get_const_corr_cov_mat(C: np.ndarray, s: float) -> np.ndarray:
    """
    Use Leidot-Wolf shrinkage estimator to compute constant correlation covariance matrix.
    """
    n = C.shape[0]

    # S is the matrix with sample standard deviations on the diagonal
    S = np.zeros_like(C)
    for i in range(n):
        S[i, i] = np.sqrt(C[i, i])
    S_inv = np.linalg.inv(S)

    R = np.matmul(S_inv,
                  np.matmul(C, S_inv))

    I = np.identity(n)
    J = np.ones((n,n))
    u = np.ones((n,))
    ut_R_u = np.matmul(np.transpose(u),
                       np.matmul(R, u))

    rho_avg = (ut_R_u - n)/(n*(n-1))
    C_rho = np.matmul(S,
                      np.matmul(I + rho_avg*(J-I), S))

    C_cc = s*C_rho + (1-s)*C
    return C_cc

if __name__ == "__main__":
    df_currencies = get_currency_data()
    print(df_currencies)
    log_returns = to_log(df_currencies)

    learning_set_log_returns, holdout_set_log_returns = learning_holdout_split(log_returns)
    print(f"Learning set period: {learning_set_log_returns.index[0]}-{learning_set_log_returns.index[-1]}")
    print(f"Holdout set period: {holdout_set_log_returns.index[0]}-{holdout_set_log_returns.index[-1]}")

    m = get_expected_returns(learning_set_log_returns)
    C = get_cov_mat(learning_set_log_returns)

    lambda1_grid = np.arange(0, 0.11, 0.01)
    mus = []
    stds = []
    efficient_portfolios = []
    for lambda1 in lambda1_grid:
        w = get_efficient_portfolio(lambda1, m, C)
        efficient_portfolios.append(w)

        mu = np.matmul(np.transpose(m), w)
        std = np.sqrt(np.matmul(np.transpose(w), np.matmul(C, w)))

        mus.append(mu)
        stds.append(std)

    # test efficient portfolios on holdout set
    C_holdout = get_cov_mat(holdout_set_log_returns)
    print(f"In-sample vs out-of-sample variance (pct/day)")
    for i in range(len(efficient_portfolios)):
        w = efficient_portfolios[i]
        var_holdout = np.matmul(np.transpose(w), np.matmul(C_holdout, w))

        print(f"{stds[i]*stds[i]:4f} vs {var_holdout:4f}")

    plt.scatter(100*np.array(stds), 10000*np.array(mus))
    plt.xlabel("Standard deviation (pct/day)")
    plt.ylabel("Return (bps/day)")
    plt.title("Markowitz Efficient Portfolios: Franc-Pound-Yen")
    plt.show()

    print("")

    # compute min. var. portfolios, based on cov. mat. and constant-corr cov mat
    w1 = get_min_variance_portfolio(C)
    w2 = get_min_variance_portfolio(get_const_corr_cov_mat(C, 1/3))

    var_w1_holdout = np.matmul(np.transpose(w1),
                               np.matmul(C_holdout, w1))
    var_w2_holdout = np.matmul(np.transpose(w2),
                               np.matmul(C_holdout, w2))
    print(f"Var of w1 on holdout set: {var_w1_holdout}")
    print(f"Var of w2 on holdout set: {var_w2_holdout}")

