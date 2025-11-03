# app.py — Final Version for Bug Prediction Dashboard
# Fully compatible with Streamlit Cloud & local runs

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Bug Prediction Dashboard", layout="wide")

# -------------------------------
# Helper Functions
# -------------------------------

def load_artifact(path):
    """Safely load model/preprocessor file or show Streamlit error."""
    if not os.path.exists(path):
        st.error(f"Missing file: {path} — please ensure it exists in the app folder.")
        st.stop()
    return joblib.load(path)

def preprocess_input(df, imputer, scaler, expected_n_features=None):
    """Impute missing values and scale features."""
    X = df.copy()

    # Convert all columns to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # Impute & scale
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled, X

# -------------------------------
# Load Artifacts
# -------------------------------
st.sidebar.title("Model Artifacts")
st.sidebar.write("Ensure these files exist:")
st.sidebar.write("- bug_predict_rf_tuned.joblib")
st.sidebar.write("- imputer.joblib")
st.sidebar.write("- scaler.joblib")
st.sidebar.write("- (optional) feature_order.txt")

try:
    model = load_artifact("bug_predict_rf_tuned.joblib")
    imputer = load_artifact("imputer.joblib")
    scaler = load_artifact("scaler.joblib")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Determine expected feature count
try:
    expected_n_features = getattr(scaler, "n_features_in_", None)
except Exception:
    expected_n_features = None

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Software Defect Prediction Dashboard")
st.markdown(
    """
    Upload a CSV file containing software metrics (same format as used in training).
    The dashboard preprocesses data, applies the trained Random Forest model,
    and displays predictions, defect probabilities, and summary insights.
    """
)

uploaded_file = st.file_uploader("Upload software metrics CSV file", type=["csv"])

# Optional sample dataset section
with st.expander("Sample dataset and guidance"):
    st.write("Ensure your CSV uses the same features and order as used during training.")
    if os.path.exists("jm1.csv"):
        with open("jm1.csv", "rb") as fh:
            st.download_button("Download sample jm1.csv", fh, file_name="jm1_sample.csv")
    else:
        st.info("No jm1.csv found locally — upload your own dataset.")

# -------------------------------
# Process Uploaded File
# -------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Unable to read CSV file. Please upload a valid comma-separated file.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --- FIX: Drop unwanted columns ---
    if 'defect' in df.columns:
        st.warning("Uploaded file contained 'defect' column — dropping it for prediction.")
        df = df.drop(columns=['defect'])

    unnamed_cols = [c for c in df.columns if 'unnamed' in c.lower()]
    if unnamed_cols:
        st.info(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # --- Handle column order (feature_order.txt) ---
    expected = None
    if os.path.exists("feature_order.txt"):
        with open("feature_order.txt") as fh:
            expected = [l.strip() for l in fh if l.strip()]

    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing expected columns: {missing}")
            st.stop()
        df = df[expected]
    else:
        if expected_n_features and df.shape[1] != expected_n_features:
            st.error(
                f"Uploaded data has {df.shape[1]} features but the model expects {expected_n_features}. "
                "Please ensure the same features are used or provide a feature_order.txt file."
            )
            st.stop()

    # --- Preprocess data ---
    X_scaled, X_original = preprocess_input(df, imputer, scaler)

    # --- Run predictions ---
    preds = model.predict(X_scaled)
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        probs = np.zeros(len(preds))

    # --- Display results ---
    results = X_original.copy()
    results["Predicted_Defect"] = preds
    results["Defect_Probability"] = np.round(probs, 4)

    st.subheader("Prediction Results (First 20 Rows)")
    st.dataframe(results.head(20))

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

    # --- Summary ---
    st.subheader("Summary")
    defect_rate = results["Predicted_Defect"].mean() * 100
    st.metric("Predicted Defect Rate", f"{defect_rate:.2f}%")

    counts = results["Predicted_Defect"].value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Top Predicted Defects (High Risk Modules)")
    topk = results.sort_values("Defect_Probability", ascending=False).head(10)
    st.dataframe(topk)

    st.info("If SHAP plots are available, they can be integrated here for explainability.")

else:
    st.info("Awaiting CSV upload... Please upload your dataset to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    """
    **Notes:**
    - The uploaded CSV must use the same features (and ideally order) as used during training.  
    - If column order differs, add a file named `feature_order.txt` with one feature per line (training order).  
    - For deployment or sharing, include this `app.py` and all `.joblib` files in your GitHub repo.  
    """
)
