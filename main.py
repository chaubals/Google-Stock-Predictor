import pandas as pd

df = pd.read_csv(
    "data/raw/googl_data_2020_2025.csv",
    header=[0,1],      # two header rows
    index_col=0        # first column is Date
)

df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

df['daily_return'] = df['Adj Close'].pct_change()
df['target_5d'] = df['Adj Close'].pct_change(5).shift(-5)

df.dropna(inplace=True)


print(df.head())