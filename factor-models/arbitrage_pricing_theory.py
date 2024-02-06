import statsmodels.api as sm
import pandas as pd
import numpy as np

FNAME_RETURN_DATA = "./data/stock_returns.csv"
FNAME_INDUSTRY_FACTOR_DATA = "./data/ValueWeighted_Monthly_5_Industry_Portfolios.csv"
FNAME_FAMAFRENCH_DATA = "./data/Monthly_3_FamaFrench_Factors.csv"

def get_return_data(fname_return_data: str) -> pd.DataFrame:
    """
    Return a TxN matrix 'R' in a labeled dataframe,
    where R[t,i] = "return of asset 'i' at time 't'""

    T = number of time periods (in months)
    N = number of assets
    """
    df_returns = pd.read_csv(fname_return_data).iloc[::-1]  # oldest to most recent
    print(f"[LOGGER]: Using returns from {df_returns.iloc[0]['Date']} " + \
          f"to {df_returns.iloc[-1]['Date']}")
    df_returns = df_returns.set_index('Date')

    company_names = df_returns.columns.tolist()
    print(f"[LOGGER]: Companies: {company_names}")

    df_returns -= 1  # return relatives to rates of return

    return df_returns

def get_factor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a tuple of DataFrames:
        - TxK matrix 'F', where F[t,j] = "factor 'j' at time 't'"
        - T-dimensional vector of 30-day T-bill rate (risk-free asset)
    """
    df_famafrench = pd.read_csv(FNAME_FAMAFRENCH_DATA)
    print(f"[LOGGER]: Using 3 Fama-French factors from " + \
          f"{df_famafrench.iloc[0]['Date']} to " + \
          f"{df_famafrench.iloc[-1]['Date']}")
    print(f"[LOGGER]: Factor names: {df_famafrench.columns}\n")

    rf = df_famafrench[['Date', 'RF']]
    F = df_famafrench.drop(columns=['RF'])
    assert(F.shape[0] == rf.shape[0])

    rf = rf.set_index('Date')
    F = F.set_index('Date')

    #  df_industry = pd.read_csv(FNAME_INDUSTRY_FACTOR_DATA)
    #  print(f"[LOGGER]: Industry data dates: {df_industry.iloc[0]['Date']} to {df_industry.iloc[-1]['Date']}.")
    #  print(f"[LOGGER]: Industries: {df_industry.columns}\n")
    #  df_industry = df_industry.set_index('Date')

    #  # join Fama-French and Industry factors
    #  F = F.join(df_industry)

    return (F, rf)

def estimate_factor_loadings(r_i: np.ndarray,
                             F: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate factor loadings ("betas") doing OLS.

    $ r_i = F\beta_i + [INTERCEPT] $
    Regression:
        - Independent variable: factors
        - Dependent variable  : excess returns
        - Parameters          : factor loadings 'beta_i'

    Return: A tuple (beta_i, intercept)
        - 'beta_i' a K-dimensional vector, where K is no. factors
        - intercept
    """
    F_tilde = sm.add_constant(F)
    assert(F_tilde.shape[0] == F.shape[0])
    assert(F_tilde.shape[1] == F.shape[1] + 1)

    model = sm.OLS(r_i, F_tilde)
    results = model.fit()
    print(f"\t\t{results.rsquared}")
    # print(results.summary())

    return results.params.drop('const'), results.params['const']

if __name__ == "__main__":
    # get dataset: returns and factors
    R = get_return_data(FNAME_RETURN_DATA)
    (F, rf) = get_factor_data()
    assert(R.shape[0] == F.shape[0])

    T = R.shape[0]  # no. time periods (in months)
    N = R.shape[1]  # no. assets
    K = F.shape[1]  # no. factors

    # define factor loadings (betas) to be calculated
    B = pd.DataFrame(index=R.columns.tolist(),
                     columns=F.columns.tolist())
    intercepts = pd.DataFrame(index=R.columns.tolist(),
                              columns=['Intercept'])

    print("[LOGGER]: Estimating factor loadings and risk premiums...")
    for asset in R.columns.tolist():
        # excess returns of asset 'i' over 'T' periods 
        r_i = R[asset] - rf['RF'].values

        print(f"\t- R^2 for {asset}:") 
        beta_i, intercept = estimate_factor_loadings(r_i, F)
        B.loc[asset] = beta_i
        intercepts.loc[asset] = intercept

    bmat = B.values.astype(float)
    B_inv = np.matmul(np.linalg.inv(np.matmul(np.transpose(bmat),
                                              bmat)),
                      np.transpose(bmat))

    print("Factor Loadings")
    print(B)
    print("\n")

    # Calculate risk premium
    risk_premiums = np.matmul(B_inv, intercepts.values.astype(float))
    df_risk_premium = pd.DataFrame(risk_premiums,
                                   index=F.columns.tolist(), columns=['Risk Prem.'])
    print(df_risk_premium)

