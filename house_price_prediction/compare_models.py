from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "House Price Prediction Dataset.csv"
TARGET = "Price"

def make_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Found: {list(df.columns)}")
    df = df.dropna(subset=[TARGET]).copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df.drop(columns=[TARGET] + (["Id"] if "Id" in df.columns else []))
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    preprocessor = make_preprocessor(X)
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        # This model accepts dense data, so use a dense encoder for this pipeline.
        "Hist Gradient Boosting": Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", SimpleImputer(strategy="median"),
                 X.select_dtypes(include="number").columns.tolist()),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), X.select_dtypes(exclude="number").columns.tolist()),
            ])),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, estimator in models.items():
        if name == "Hist Gradient Boosting":
            pipeline = estimator
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])
        scores = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            n_jobs=-1,
        )
        rows.append({
            "Model": name,
            "Mean R2": round(scores["test_r2"].mean(), 4),
            "Mean MAE": round(-scores["test_mae"].mean(), 2),
        })

    results = pd.DataFrame(rows).sort_values("Mean R2", ascending=False)
    print("\nCross-validation results (higher R2 and lower MAE are better):\n")
    print(results.to_string(index=False))
    output_path = ROOT / "model_comparison.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()
