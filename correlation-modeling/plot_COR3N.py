import matplotlib.pyplot as plt
import pandas as pd

# DATA_URL = "https://www.cboe.com/7bff1674-afde-486a-a78f-460b779ae046"
FNAME = "cor3m-history.csv"

if __name__ == "__main__":
    df = pd.read_csv(FNAME)
    for col_name in df.columns.tolist()[1:]:  # skip date
        plt.plot(range(len(df)), df[col_name])
        plt.grid()
        plt.title(f'CBOE implied correlation {col_name}')
        nobs = len(df)
        stride = 10
        plt.xticks(range(0, nobs, stride), df["Date"].iloc[0:nobs:stride])
        plt.xlim(0, nobs)
        plt.show()

