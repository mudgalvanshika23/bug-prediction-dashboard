# app.py — Final Version (Prediction + Evaluation + Visualization Dashboard)
# Software Defect Prediction using Random Forest Model (JM1 Dataset)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Software Defect Prediction Dashboard", layout="wide")

# -------------------------------
# Helper Functions
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
# Sidebar – Model Info
# -------------------------------
st.sidebar.title("Model Artifacts")
st.sidebar.write("Ensure these files exist in the app folder:")
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
# Main Header
# -------------------------------
st.title("Software Defect Prediction Dashboard")
st.markdown(
    """
    This dashboard predicts and evaluates defect-prone software modules using a trained Random Forest model.  
    You can upload:
    - A CSV **without** the `defect` column for prediction.  
    - A CSV **with** the `defect` column for evaluation.
    """
)

uploaded_file = st.file_uploader("Upload software metrics CSV file", type=["csv"])

# -------------------------------
# Optional Sample Dataset
# -------------------------------
with st.expander("Sample Dataset and Format Guide"):
    st.write("Ensure your CSV uses the same structure as the JM1 dataset used for model training.")
    if os.path.exists("jm1.csv"):
        with open("jm1.csv", "rb") as fh:
            st.download_button("Download sample jm1.csv", fh, file_name="jm1_sample.csv")
    else:
        st.info("No jm1.csv found locally — upload your own dataset.")

# -------------------------------
# Process Uploaded CSV
# -------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Unable to read CSV file. Please upload a valid file.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Drop unnamed columns ---
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        st.info(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # --- Optional correlation heatmap ---
    with st.expander("Show Correlation Heatmap"):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax)
        st.pyplot(fig)

    # --- Label handling ---
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

    # --- Handle column order using feature_order.txt ---
    expected = None
    if os.path.exists("feature_order.txt"):
        with open("feature_order.txt") as fh:
            expected = [l.strip() for l in fh if l.strip()]
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            st.error(f"Uploaded file missing expected columns: {missing}")
            st.stop()
        df = df[expected]
    else:
        if expected_n_features and df.shape[1] != expected_n_features:
            st.error(
                f"Uploaded data has {df.shape[1]} features but model expects {expected_n_features}."
            )
            st.stop()

    # --- Preprocess & Predict ---
    X_scaled, X_original = preprocess_input(df, imputer, scaler)
    probs = model.predict_proba(X_scaled)[:, 1]
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    preds = (probs >= threshold).astype(int)

    # --- Results ---
    results = X_original.copy()
    results["Predicted_Defect"] = preds
    results["Defect_Probability"] = np.round(probs, 4)

    st.subheader("Prediction Results (First 20 Rows)")
    st.dataframe(results.head(20))

    # --- Summary statistics ---
    st.subheader("Summary Statistics")
    defect_rate = results["Predicted_Defect"].mean() * 100
    st.metric("Predicted Defect Rate", f"{defect_rate:.2f}%")
    counts = results["Predicted_Defect"].value_counts().sort_index()
    st.bar_chart(counts)

    # --- Feature Importance ---
    try:
        importances = model.feature_importances_
        feat_names = X_original.columns.tolist()
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        st.subheader("Feature Importances (Random Forest)")
        st.bar_chart(fi.head(20))
    except Exception as e:
        st.info("Feature importance not available: " + str(e))

    # --- Top High-Risk Modules ---
    st.subheader("Top Predicted Defects (High-Risk Modules)")
    k = st.slider("Top K modules to display", 5, 50, 10, 1)
    topk = results.sort_values("Defect_Probability", ascending=False).head(k)
    st.dataframe(topk)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

    # -------------------------------
    # Evaluation Mode
    # -------------------------------
    if eval_mode:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score,
            roc_curve, precision_recall_curve
        )

        st.subheader("Model Evaluation (Using Provided Labels)")

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
            st.write(f"**ROC AUC:** {roc_auc:.4f}")

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC & Precision-Recall Curves
        fpr, tpr, _ = roc_curve(y_true, probs)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, probs)

        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}" if roc_auc else "")
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(pr_rec, pr_prec)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        st.pyplot(fig2)

    # --- Optional SHAP Visualization ---
    if os.path.exists("shap_summary.png"):
        st.subheader("SHAP Summary (Global Feature Impact)")
        img = Image.open("shap_summary.png")
        st.image(img, use_column_width=True)

else:
    st.info("Awaiting CSV upload... Please upload your dataset to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    """
    **Notes:**
    - You can upload CSV files with or without the `defect` column.  
    - Use the threshold slider to adjust defect classification sensitivity.  
    - If a `feature_order.txt` exists, it ensures consistent feature alignment.  
    - Evaluation mode shows confusion matrix, ROC, and PR curves.  
    - SHAP summary plot can be added as `shap_summary.png` for explainability.  
    """
)
