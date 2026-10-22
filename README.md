# 🏠 House Price Prediction

A machine-learning project that trains a regression model on the included housing CSV and provides a Streamlit web app for interactive predictions.

## Dataset
The included `House Price Prediction Dataset.csv` contains 2,000 rows and these columns:

- `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`
- `Location`, `Condition`, `Garage`
- Target: `Price`
- `Id` is excluded from training because it is an identifier.

## Tech stack
Python, Pandas, Scikit-learn, Random Forest Regression, Joblib, Streamlit.

## Run locally

### 1. Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This creates `artifacts/house_price_model.joblib` and `artifacts/metrics.json`.

### 4. Launch the web app
```bash
streamlit run app.py
```
Open the local URL shown in your terminal, usually `http://localhost:8501`.

## Evaluation
The training script reports:
- **MAE**: average absolute prediction error, in the target's units.
- **RMSE**: error metric that penalizes larger mistakes more heavily.
- **R²**: goodness-of-fit score on the held-out test set. It can be negative if the model performs worse than predicting the test-set mean.

## Important notes
- The model's predictions are only as reliable as the supplied dataset.
- The CSV does not document the currency or units for `Price` and `Area`; do not label the output as INR, square feet, etc. until those units are confirmed.
- A random 80/20 split is used. For a serious real-estate application, validate against newer or geographically separate data and compare multiple models.
- Retrain after replacing or updating the dataset.
