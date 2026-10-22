from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nNumeric summary:\n", df.describe(include="all").T)

if "Price" not in df.columns:
    raise ValueError(f"Expected target column 'Price'. Found: {list(df.columns)}")

numeric = df.select_dtypes(include="number")
print("\nNumeric correlations with Price:\n")
print(numeric.corr(numeric_only=True)["Price"].sort_values(ascending=False))

plt.figure()
df["Price"].dropna().hist(bins=30)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(ROOT / "price_distribution.png", dpi=150)
print("\nSaved price_distribution.png")
