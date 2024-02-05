import pandas as pd
import numpy as np


FNAME_MARKETWIDE_FACTORS_DATA = "./data/annual_F-F_Research_Data_Factors.CSV"
FNAME_5_INDUSTRY_FACTORS_DATA = "./data/annual_value_weighted_5_Industry_Portfolios.CSV"
FNAME_RETURN_DATA = "./data/ratio_data.csv"


def get_factors() -> np.ndarray:
    """
    Return log-returns on factors.
    """

    def to_log_return(r):
        """5.0 = 5% is transformed to log(1.05)"""
        return np.log(1+r/100)

    df_marketwide_factors = pd.read_csv(FNAME_MARKETWIDE_FACTORS_DATA).drop(columns=['RF']).tail(1)
    marketwide_factor_names = df_marketwide_factors.columns.tolist()[1:]  # exclude date
    print(f"Marketwide factors: {marketwide_factor_names}")
    df_marketwide_factors[marketwide_factor_names] = df_marketwide_factors[marketwide_factor_names].apply(to_log_return)

    df_industry_factors = pd.read_csv(FNAME_5_INDUSTRY_FACTORS_DATA).tail(1)
    industry_factor_names = df_industry_factors.columns.tolist()[1:]  # exclude date
    print(f"Industry factors: {industry_factor_names}")
    df_industry_factors[industry_factor_names] = df_industry_factors[industry_factor_names].apply(to_log_return)

    print(df_marketwide_factors)
    print(df_industry_factors)

    # df_factors = df_marketwide_factors.join(df_industry_factors, on='Date')
    df_factors = pd.merge(df_marketwide_factors, df_industry_factors, on='Date')
    print(df_factors)

    factors = df_factors[df_factors.columns.tolist()[1:]].values.squeeze().astype(float)
    assert(factors.shape == (8,))

    # 5 industry data has -99.99 or -999 for missing data
    return factors 

def get_returns() -> np.ndarray:
    df_returns = pd.read_csv(FNAME_RETURN_DATA).head(12).iloc[::-1]  # 2023 monthly
    print(df_returns)

    monthly_return_relatives = df_returns.values[:, 1:].astype(float)  # drop dates
    return_relatives = np.prod(monthly_return_relatives, axis=0)
    assert(return_relatives.shape == (11,))

    log_returns = np.log(return_relatives)
    return log_returns


if __name__ == "__main__":
    X = get_factors()
    Y = get_returns()

