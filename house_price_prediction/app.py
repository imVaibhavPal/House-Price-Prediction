from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "House Price Prediction Dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "house_price_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Enter the house details below to estimate its price using the model trained on your dataset.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` in the terminal first.")
    st.stop()
if not DATA_PATH.exists():
    st.error("Dataset CSV is missing from the project folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
df = pd.read_csv(DATA_PATH)
feature_columns = [c for c in df.columns if c not in ("Price", "Id")]

with st.form("prediction_form"):
    st.subheader("House details")
    values = {}

    cols = st.columns(2)
    for i, column in enumerate(feature_columns):
        series = df[column]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(series):
                median = float(series.median())
                min_value = float(series.min())
                max_value = float(series.max())
                if min_value == max_value:
                    values[column] = st.number_input(column, value=median)
                else:
                    values[column] = st.number_input(
                        column,
                        min_value=min_value,
                        max_value=max_value,
                        value=median,
                        step=1.0 if series.dropna().mod(1).eq(0).all() else 0.1,
                    )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    options = ["Unknown"]
                values[column] = st.selectbox(column, options)

    submitted = st.form_submit_button("Predict house price", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=feature_columns)
    prediction = float(model.predict(input_df)[0])
    st.success(f"Estimated house price: {prediction:,.2f}")
    st.caption("This is a model estimate, not a guaranteed market price. Predictions depend on the quality and coverage of the training data.")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Model evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:,.2f}")
    c2.metric("RMSE", f"{metrics['rmse']:,.2f}")
    c3.metric("R² score", f"{metrics['r2']:.3f}")
    st.caption(f"Evaluated on {metrics['test_rows']} held-out rows. Metrics describe this dataset and split, not guaranteed real-world performance.")
