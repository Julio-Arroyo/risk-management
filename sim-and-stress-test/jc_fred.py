from typing import List
import pandas as pd
import numpy as np
import fredapi


def to_log(df_currencies: pd.DataFrame) -> pd.DataFrame:
    df_log_currencies = pd.DataFrame()
    for exchange_rate_name in df_currencies.columns.to_list():
        log_er_name = f"LOG({exchange_rate_name})"

        today_rate = df_currencies[exchange_rate_name].iloc[0:-1]
        tomorrow_rate = df_currencies[exchange_rate_name].iloc[1:]
        log_returns = np.log(tomorrow_rate.values/today_rate.values)

        df_log_currencies[log_er_name] = pd.Series(log_returns,
                                                   index=tomorrow_rate.index.tolist())
    # print(df_log_currencies)
    return df_log_currencies


def get_FRED_data(series_names: List[str],
                  labels: List[str],
                  inverted: List[bool],
                  obs_end: str,
                  convert_log=True) -> pd.DataFrame:
    """
    Return a DataFrame of data for the series in 'series_names'

    Parameters:
    - labels: column names for DataFrame
    - inverted: if inverted[i]=TRUE then get 1/x_i instead of x_i
    """
    fred = fredapi.Fred(api_key='8fa3802ea771fad2cef32f4f625e7907')

    df_currencies = pd.DataFrame()
    for i in range(len(series_names)):
        currency_series = fred.get_series(series_names[i],
                                          observation_start=None,
                                          observation_end=obs_end)
        # make sure dataframe has data in USD/currency
        if inverted[i]:
            df_currencies[labels[i]] = currency_series.apply(lambda x: 1/x)
        else:
            assert(not inverted[i])
            df_currencies[labels[i]] = currency_series

    df_currencies = df_currencies.dropna()  # drop dates with at least one NaN
    if convert_log:
        df_currencies = to_log(df_currencies)
    return df_currencies

