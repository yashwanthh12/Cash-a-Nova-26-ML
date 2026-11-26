"""Train a Random Forest model to predict shipping delay in days."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("Dataset.csv")
RANDOM_STATE = 42

LEAKAGE_AND_ID_COLUMNS = [
    "Days for shipping (real)", "Late_delivery_risk", "Order ID",
    "Customer ID", "Product Card Id", "Product Category Id",
    "Product Image", "Product Name", "Product Description", "Product Status"
]

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            "Dataset.csv not found. Put it beside train_model.py."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalize inconsistent category spelling/capitalization.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (df[col].astype("string").str.strip().str.casefold()
                   .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA}))

    if "Shipping Delay" in df.columns:
        y = pd.to_numeric(df["Shipping Delay"], errors="coerce")
    elif {"Days for shipping (real)", "Days for shipment (scheduled)"} <= set(df.columns):
        actual = pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
        scheduled = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        y = actual - scheduled
    else:
        raise ValueError(
            "Need 'Shipping Delay', or both actual and scheduled shipping-day columns."
        )

    if "Order Item Quantity" in df.columns:
        qty = pd.to_numeric(df["Order Item Quantity"], errors="coerce")
        scheduled = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        # A simple engineered proxy, not a measured operational variable.
        df["scheduling_pressure"] = np.log1p(qty.clip(lower=0)) / (
            scheduled.clip(lower=0) + 1
        )

    X = df.drop(columns=["Shipping Delay"], errors="ignore")
    X = X.drop(columns=LEAKAGE_AND_ID_COLUMNS, errors="ignore")
    valid = y.notna()
    return X.loc[valid].copy(), y.loc[valid].astype(float)

def build_pipeline(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    forest = RandomForestRegressor(
        n_estimators=300, max_depth=15,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    return Pipeline([("preprocessing", preprocess), ("model", forest)])

def main():
    X, y = load_data(DATA_PATH)
    print(f"Rows: {len(X):,} | Input columns: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n--- Held-out test performance ---")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f} days")
    print(f"R²:  {r2_score(y_test, predictions):.3f}")

    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    print("\n--- Top input features ---")
    print(importance.head(10).to_string(index=False))
    importance.to_csv("feature_importance.csv", index=False)

    joblib.dump(model, "shipping_delay_model.joblib")
    print("\nSaved shipping_delay_model.joblib and feature_importance.csv")

if __name__ == "__main__":
    main()
