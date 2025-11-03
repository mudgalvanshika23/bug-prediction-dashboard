# app.py — Stable Version (Working on Streamlit Cloud)
# Software Defect Prediction Dashboard – Core Functionality

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Software Defect Prediction Dashboard", layout="wide")

# -------------------------------
# Helper functions
# -------------------------------
def load_artifact(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

def preprocess_input(df, imputer, scaler):
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled, X

# -------------------------------
# Sidebar – Load Artifacts
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
    st.error(f"Error loading artifacts: {e}")
    st.stop()

try:
    expected_n_features = getattr(scaler, "n_features_in_", None)
except Exception:
    expected_n_features = None

# -------------------------------
# Header
# -------------------------------
st.title("Software Defect Prediction Dashboard")
st.markdown(
    """
    Upload a CSV file with software metrics to predict defect-prone modules.  
    You can also upload the dataset **with the 'defect' column** to evaluate model accuracy.
    """
)

uploaded_file = st.file_uploader("Upload your software metrics CSV file", type=["csv"])

# -------------------------------
# Sample Data
# -------------------------------
with st.expander("Sample dataset and guidance"):
    st.write("Use the same column structure as JM1 dataset used in training.")
    if os.path.exists("jm1.csv"):
        with open("jm1.csv", "rb") as fh:
            st.download_button("Download sample jm1.csv", fh, file_name="jm1_sample.csv")
    else:
        st.info("No jm1.csv found locally — upload your own dataset.")

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error reading CSV file.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        st.info(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    has_label = "defect" in df.columns
    eval_mode = False
    if has_label:
        choice = st.radio(
            "Detected 'defect' column. Choose action:",
            ("Evaluate model using provided labels", "Drop labels and only predict"),
        )
        if choice == "Evaluate model using provided labels":
            eval_mode = True
            y_true = df["defect"].astype(int).values
            df = df.drop(columns=["defect"])
        else:
            st.warning("Dropping 'defect' column for prediction.")
            df = df.drop(columns=["defect"])

    expected = None
    if os.path.exists("feature_order.txt"):
        with open("feature_order.txt") as fh:
            expected = [l.strip() for l in fh if l.strip()]
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
            st.stop()
        df = df[expected]
    else:
        if expected_n_features and df.shape[1] != expected_n_features:
            st.error(f"Uploaded data has {df.shape[1]} features; model expects {expected_n_features}.")
            st.stop()

    # Preprocess & Predict
    X_scaled, X_original = preprocess_input(df, imputer, scaler)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))

    results = X_original.copy()
    results["Predicted_Defect"] = preds
    results["Defect_Probability"] = np.round(probs, 4)

    st.subheader("Prediction Results (First 20 Rows)")
    st.dataframe(results.head(20))

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions (CSV)",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Summary
    st.subheader("Summary")
    defect_rate = results["Predicted_Defect"].mean() * 100
    st.metric("Predicted Defect Rate", f"{defect_rate:.2f}%")
    st.bar_chart(results["Predicted_Defect"].value_counts().sort_index())

    st.subheader("Top Predicted Defects (High-Risk Modules)")
    st.dataframe(results.sort_values("Defect_Probability", ascending=False).head(10))

    # Evaluation
    if eval_mode:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        st.subheader("Evaluation Results (Using Provided Labels)")
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, probs)
        except Exception:
            roc_auc = None

        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")
        if roc_auc:
            st.write(f"**ROC-AUC:** {roc_auc:.4f}")

else:
    st.info("Awaiting CSV upload... Please upload your dataset to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    """
    **Notes:**
    - Upload a CSV file with software metrics for predictions.  
    - If the dataset includes the `defect` column, select 'Evaluate' to view performance metrics.  
    - All predictions are downloadable as CSV for further analysis.  
    """
)
