import numpy as np
from src.data_loader import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt

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

df = master_df.copy()

#Google features
df["goog_ret_1"] = df["Adj Close"].pct_change()
df["goog_ret_5"] = df["Adj Close"].pct_change(5)
df["ma_20"] = df["Adj Close"].rolling(20).mean()
df["vol_20"] = df["Adj Close"].rolling(20).std()

#Market returns
df["nasdaq_ret"] = df["NASDAQ"].pct_change()
df["sp500_ret"] = df["SP500"].pct_change()

#Macro changes
df["cpi_change"] = df["CPI"].pct_change()
df["gdp_change"] = df["GDP"].pct_change()
df["unemp_change"] = df["UNRATE"].diff()
df["vix_change"] = df["VIX"].pct_change()
df["treasury_change"] = df["Treasury"].diff()

df["target_5d"] = df["Adj Close"].pct_change(5).shift(-5)

df.dropna(inplace=True)

feature_cols = [
  "goog_ret_1",
  "goog_ret_5",
  "ma_20",
  "vol_20",
  "nasdaq_ret",
  "sp500_ret",
  "cpi_change",
  "gdp_change",
  "unemp_change",
  "vix_change",
  "treasury_change"
]

x = df[feature_cols]
y = df["target_5d"]

# Train/Test split for data - splitting data on basis of date (time-based)

split_date = "2024-01-01"

x_train = x[x.index < split_date]
x_test = x[x.index >= split_date]

y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]

# Standardize data

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Baseline linear regression
lin_model = LinearRegression()
lin_model.fit(x_train_scaled, y_train)

lin_preds = lin_model.predict(x_test_scaled)

print("Linear R^2:", lin_model.score(x_test_scaled, y_test))

# SVR (Multvariate Model)
svr = SVR(kernel="rbf", C=1.0, epsilon=0.01)
svr.fit(x_train_scaled, y_train)

svr_preds = svr.predict(x_test_scaled)

plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, svr_preds, label="SVR Predicted")
plt.legend()
plt.title("5-Day Forward Return Prediction")
plt.show()

# print(df.shape)
# print(df.head())

# print(master_df.head())
# print(master_df.tail())
# print(master_df.shape)

# print("GOOG:", goog.index.min(), "to", goog.index.max())
# print("NASDAQ:", nasdaq.index.min(), "to", nasdaq.index.max())
# print("SP500:", sp500.index.min(), "to", sp500.index.max())
# print("CPI:", cpi.index.min(), "to", cpi.index.max())
# print("GDP:", gdp.index.min(), "to", gdp.index.max())
# print("UNEMP:", unemp.index.min(), "to", unemp.index.max())
# print("VIX:", vix.index.min(), "to", vix.index.max())
# print("Treasury:", treasury.index.min(), "to", treasury.index.max())
