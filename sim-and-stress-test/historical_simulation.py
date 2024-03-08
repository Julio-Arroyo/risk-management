"""
Author: Julio Arroyo 2024

Perform historical simulation on a portfolio of assets.
"""


import jc_statistics as jc_stats
import pandas as pd
import numpy as np
import scipy.stats
import jc_fred
import math


def compute_statistics(log_returns: np.ndarray, asset_name) -> pd.Series:
    T = len(log_returns)

    statistic_names = ['Min', 'Max',  # extreme values
                       'Mean', 'Volatility', 'Skewness', 'Excess Kurtosis',  # central moments
                       'Pr(normality)',
                       '99% VaR', '99% cVaR',  # Risk metrics
                       'Serial Correlation',
                       'Cumulative Return', 'Annualized Return']  # Performance
    statistic_values = pd.Series(np.array([None for _ in range(len(statistic_names))]),
                                 index=statistic_names,
                                 name=asset_name)
    
    statistic_values['Min'] = np.min(log_returns)
    statistic_values['Max'] = np.max(log_returns)

    statistic_values['Mean'] = np.mean(log_returns)
    statistic_values['Volatility'] = np.std(log_returns)
    statistic_values['Skewness'] = jc_stats.skewness(log_returns)
    statistic_values['Excess Kurtosis'] = jc_stats.kurtosis(log_returns)
    statistic_values['Pr(normality)'] = jc_stats.jarque_bera_test(log_returns)[1]
    
    sorted_log_returns = np.sort(log_returns)
    one_percentile_idx = math.ceil(0.01*(T+1))
    statistic_values['99% VaR'] = -sorted_log_returns[one_percentile_idx]
    statistic_values['99% cVaR'] = -np.mean(sorted_log_returns[:(one_percentile_idx + 1)])
    
    statistic_values['Serial Correlation'] = scipy.stats.pearsonr(log_returns[:-1], log_returns[1:])[0]

    return statistic_values


def simulate_history(asset_log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
    - asset_log_returns: TxN matrix. T='no. obs.' and N='no. assets'

    Return:
    - portfolio_stats: DataFrame w/ metrics on portfolio
        > Extreme values:
            * Min
            * Max
        > Central moments:
            * Mean
            * Standard Deviation
            * Skewness (third moment)
            * Excess Kurtosis (fourth moment - 3)
            * Pr(portfolio returns are log-normally distributed)
        > Risk Metrics:
            * 99% Value-at-Risk
            * 99% conditional Value-at-Risk
        > Serial correlation:
        > Performance:
            * Cumulative Return
            * Annualized Return
    """
    X = asset_log_returns.values
    T, N = X.shape

    X_returns = np.exp(X)
    portfolio_returns = np.mean(X_returns, axis=1)  # equally-weighted portfolio
    assert(portfolio_returns.shape[0] == T)
    log_portfolio_returns = np.log(portfolio_returns)

    list_stats = [compute_statistics(log_portfolio_returns, "Equally-weighted Portfolio")]
    for asset_name in asset_log_returns.columns.to_list():
        asset_stats = compute_statistics(asset_log_returns[asset_name].values,
                                         asset_name)
        list_stats.append(asset_stats)

    historical_sim = pd.concat(list_stats, axis=1)
    return historical_sim


if __name__ == "__main__":
    SERIES_NAMES = ['DEXSZUS',  # Swiss Francs to One US dollar
                    'DEXUSUK',  # US dollars to One UK Pound Sterling
                    'DEXJPUS']  # Japanese Yen to One US dollar
    inverted = [True, False, True]
    LAST_YEAR_END = '2023-12-29'
    LABELS = ['USD/Swissies', 'USD/Pounds', 'USD/Yen']
    
    asset_log_returns = jc_fred.get_FRED_data(SERIES_NAMES, LABELS, inverted, LAST_YEAR_END)
    historical_sim = simulate_history(asset_log_returns)
    print(historical_sim)

