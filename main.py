import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Google features
df["goog_ret_1"] = df["Adj Close"].pct_change()
df["goog_ret_5"] = df["Adj Close"].pct_change(5)
df["ma_20"] = df["Adj Close"].rolling(20).mean()
df["vol_20"] = df["Adj Close"].rolling(20).std()

# Lagged GOOG daily returns (past values as predictors — standard for time series)
df["goog_ret_lag1"] = df["goog_ret_1"].shift(1)
df["goog_ret_lag2"] = df["goog_ret_1"].shift(2)
df["goog_ret_lag3"] = df["goog_ret_1"].shift(3)

# Market returns
df["nasdaq_ret"] = df["NASDAQ"].pct_change()
df["sp500_ret"] = df["SP500"].pct_change()

# Macro changes
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
    "goog_ret_lag1",
    "goog_ret_lag2",
    "goog_ret_lag3",
    "ma_20",
    "vol_20",
    "nasdaq_ret",
    "sp500_ret",
    "cpi_change",
    "gdp_change",
    "unemp_change",
    "vix_change",
    "treasury_change",
]

x = df[feature_cols]
y = df["target_5d"]

# Train/Test split for data - splitting data on basis of date (time-based)

split_date = "2024-01-01"

x_train = x[x.index < split_date]
x_test = x[x.index >= split_date]

y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]

# Feature scaling: fit on train only (required for SVR; helps linear models too)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


def regression_metrics(y_true, y_pred):
    """R², RMSE, MAE, and directional accuracy (sign of predicted vs actual return)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    directional_acc = float(np.mean((y_pred > 0) == (y_true > 0)))
    return r2, rmse, mae, directional_acc


def metrics_row(model_name, y_true, y_pred, extra=None):
    r2, rmse, mae, dir_acc = regression_metrics(y_true, y_pred)
    pred_std = float(np.std(np.asarray(y_pred)))
    row = {
        "Model": model_name,
        "R^2": round(r2, 4),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Dir_acc": round(dir_acc, 4),
        "Pred_std_test": round(pred_std, 6),
    }
    if extra:
        row.update(extra)
    return row


def pick_svr_with_variability(svr_search, X_train, y_train, rel_slack=0.02):
    """
    Among CV scores within rel_slack of the best neg-MSE, choose the fitted SVR whose
    train predictions have the highest std. Reduces nearly-flat predictors from huge
    epsilon + tiny C while keeping CV error near-optimal.
    """
    best_score = svr_search.best_score_
    slack = max(1e-12, rel_slack * abs(best_score))
    scores = svr_search.cv_results_["mean_test_score"]
    params_list = svr_search.cv_results_["params"]
    near_best = [i for i, s in enumerate(scores) if s >= best_score - slack]
    best_train_std = -np.inf
    chosen_params = None
    for i in near_best:
        p = params_list[i]
        m = SVR(kernel="rbf", **p)
        m.fit(X_train, y_train)
        train_std = float(np.std(m.predict(X_train)))
        if train_std > best_train_std:
            best_train_std = train_std
            chosen_params = p
    final = SVR(kernel="rbf", **chosen_params)
    final.fit(X_train, y_train)
    return final, chosen_params, best_train_std


# Baselines: OLS + regularized linear models (multicollinearity / overfitting)
lin_model = LinearRegression()
lin_model.fit(x_train_scaled, y_train)
lin_preds = lin_model.predict(x_test_scaled)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(x_train_scaled, y_train)
ridge_preds = ridge.predict(x_test_scaled)

lasso = Lasso(alpha=1e-4, max_iter=50_000, random_state=42)
lasso.fit(x_train_scaled, y_train)
lasso_preds = lasso.predict(x_test_scaled)

# SVR: constrained grid — avoid tiny C and large epsilon (common cause of flat predictions).
# CV still optimizes MSE; tie-breaker prefers more variable train predictions among near-winners.
svr_param_grid = {
    "C": [1.0, 5.0, 10.0, 25.0],
    "epsilon": [0.005, 0.01, 0.02, 0.05],
    "gamma": ["scale", "auto"],
}
tscv = TimeSeriesSplit(n_splits=5)
svr_search = GridSearchCV(
    SVR(kernel="rbf"),
    svr_param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
svr_search.fit(x_train_scaled, y_train)
svr_model, svr_chosen_params, svr_train_pred_std = pick_svr_with_variability(
    svr_search, x_train_scaled, y_train, rel_slack=0.02
)

svr_preds = svr_model.predict(x_test_scaled)

results_rows = [
    metrics_row("Linear regression", y_test, lin_preds),
    metrics_row("Ridge (alpha=1.0)", y_test, ridge_preds),
    metrics_row("Lasso (alpha=1e-4)", y_test, lasso_preds),
    metrics_row(
        "SVR (RBF, tuned)",
        y_test,
        svr_preds,
        extra={
            "SVR_C": svr_chosen_params["C"],
            "SVR_epsilon": svr_chosen_params["epsilon"],
            "SVR_gamma": svr_chosen_params["gamma"],
        },
    ),
]
results_df = pd.DataFrame(results_rows)

# Wide table for the report + compact table without SVR hyperparameter columns duplicated
display_cols = ["Model", "R^2", "RMSE", "MAE", "Dir_acc", "Pred_std_test"]
print("\n=== Test-set metrics (holdout from split_date) ===\n")
print(results_df[display_cols].to_string(index=False))

svr_row = results_df[results_df["Model"] == "SVR (RBF, tuned)"].iloc[0]
print(
    f"\nSVR selected: C={svr_row['SVR_C']}, epsilon={svr_row['SVR_epsilon']}, "
    f"gamma={svr_row['SVR_gamma']}"
)
print(
    f"(GridSearchCV best CV neg-MSE: {svr_search.best_score_:.6f}; "
    f"raw best params: {svr_search.best_params_}; "
    f"train pred std after tie-break: {svr_train_pred_std:.6f})"
)

out_dir = Path(__file__).resolve().parent / "results"
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "model_comparison.csv"
results_df[display_cols].to_csv(csv_path, index=False)
print(f"\nSaved (report table): {csv_path}")
params_path = out_dir / "svr_selected_params.txt"
with open(params_path, "w", encoding="utf-8") as f:
    f.write(
        f"C={svr_row['SVR_C']}, epsilon={svr_row['SVR_epsilon']}, gamma={svr_row['SVR_gamma']}\n"
    )
print(f"Saved (SVR hyperparameters): {params_path}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, svr_preds, label="SVR predicted (constrained + tie-break)")
plt.legend()
plt.title("5-Day Forward Return Prediction")
plt.show()
