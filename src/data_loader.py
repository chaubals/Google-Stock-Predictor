import pandas as pd

# --------------------------------------------------
# Load FRED-style CSV (CPI, GDP, UNRATE, VIX, DGS10)
# --------------------------------------------------
def load_fred_csv(path, value_col, new_name):
    df = pd.read_csv(path)

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df.set_index("observation_date", inplace=True)
    df.sort_index(inplace=True)

    df = df[[value_col]]
    df.columns = [new_name]

    return df


# --------------------------------------------------
# Load Yahoo Finance CSV (GOOG)
# --------------------------------------------------
def load_yahoo_csv(path):
    df = pd.read_csv(
        path,
        header=[0, 1],
        index_col=0
    )

    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)

    return df

# --------------------------------------------------
# Resample to Daily
# --------------------------------------------------
def resample_to_daily(df):
    return df.resample("D").ffill()


# --------------------------------------------------
# Merge Everything
# --------------------------------------------------
def merge_datasets(base_df, datasets):
    df = base_df.copy()
    for dataset in datasets:
        df = df.join(dataset, how="inner")
    return df
