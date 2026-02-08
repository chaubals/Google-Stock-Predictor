from src.data_loader import *

# Load GOOG
goog = load_yahoo_csv("data/raw/goog.csv")

# Load FRED datasets
cpi = load_fred_csv("data/raw/CPI.csv", "CPIAUCSL", "CPI")
gdp = load_fred_csv("data/raw/GDP.csv", "GDP", "GDP")
unemp = load_fred_csv("data/raw/unemployment.csv", "UNRATE", "UNRATE")
vix = load_fred_csv("data/raw/VIX.csv", "VIXCLS", "VIX")
treasury = load_fred_csv("data/raw/treasury.csv", "DGS10", "Treasury")
nasdaq = load_fred_csv("data/raw/NASDAQ.csv", "NASDAQCOM", "NASDAQ")
sp500 = load_fred_csv("data/raw/SP500.csv", "SP500", "SP500")

# Resample lower-frequency data
cpi = resample_to_daily(cpi)
gdp = resample_to_daily(gdp)
unemp = resample_to_daily(unemp)

# Merge
master_df = merge_datasets(
    goog,
    [nasdaq, sp500, cpi, gdp, unemp, vix, treasury]
)

print(master_df.head())
print(master_df.tail())
print(master_df.shape)

# print("GOOG:", goog.index.min(), "to", goog.index.max())
# print("NASDAQ:", nasdaq.index.min(), "to", nasdaq.index.max())
# print("SP500:", sp500.index.min(), "to", sp500.index.max())
# print("CPI:", cpi.index.min(), "to", cpi.index.max())
# print("GDP:", gdp.index.min(), "to", gdp.index.max())
# print("UNEMP:", unemp.index.min(), "to", unemp.index.max())
# print("VIX:", vix.index.min(), "to", vix.index.max())
# print("Treasury:", treasury.index.min(), "to", treasury.index.max())
