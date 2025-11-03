# app.py — Final Version (Prediction + Evaluation Mode)
# Streamlit dashboard for Software Defect Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Software Defect Prediction Dashboard", layout="wide")

# -------------------------------
# Helper Functions
# -------------------------------
def load_artifact(path):
    """Safely load a model/preprocessor file."""
    if not os.path.exists(path):
        st.error(f"Missing file: {path} — please ensure it exists in the app folder.")
        st.stop()
    return joblib.load(path)

def preprocess_input(df, imputer, scaler):
    """Convert to numeric, impute missing values, and scale features."""
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled, X

# -------------------------------
# Load Model and Preprocessors
# -------------------------------
st.sidebar.title("Model Artifacts")
st.sidebar.write("Ensure these files are present:")
st.sidebar.write("- bug_predict_rf_tuned.joblib (model)")
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

# Determine expected feature count from scaler
try:
    expected_n_features = getattr(scaler, "n_features_in_", None)
except Exception:
    expected_n_features = None

# -------------------------------
# Streamlit Header and Description
# -------------------------------
st.title("Software Defect Prediction Dashboard")
st.markdown(
    """
    This dashboard predicts defect-prone software modules using a trained Random Forest model.
    You can upload:
    - A CSV **without** the `defect` column for prediction.
    - A CSV **with** the `defect` column for evaluation of model performance.
    """
)

uploaded_file = st.file_uploader("Upload software metrics CSV file", type=["csv"])

# -------------------------------
# Optional: Provide Sample Dataset
# -------------------------------
with st.expander("Sample dataset and guidance"):
    st.write("Ensure your CSV follows the same format as used during training.")
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
        st.error("Unable to read CSV. Please upload a valid comma-separated file.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Drop unnamed index columns
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        st.info(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # Check for 'defect' column and ask how to proceed
    has_label = "defect" in df.columns
    eval_mode = False
    if has_label:
        choice = st.radio(
            "Detected 'defect' column. Choose action:",
            ("Evaluate model using provided labels", "Drop labels and only predict")
        )
        if choice == "Evaluate model using provided labels":
            eval_mode = True
            y_true = df["defect"].astype(int).values
            df = df.drop(columns=["defect"])
        else:
            st.warning("Dropping 'defect' column for prediction.")
            df = df.drop(columns=["defect"])

    # Handle column order using feature_order.txt
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
                f"Uploaded data has {df.shape[1]} features, but the model expects {expected_n_features}. "
                "Ensure correct features or include a feature_order.txt file."
            )
            st.stop()

    # Preprocess and Predict
    X_scaled, X_original = preprocess_input(df, imputer, scaler)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))

    # Results table
    results = X_original.copy()
    results["Predicted_Defect"] = preds
    results["Defect_Probability"] = np.round(probs, 4)

    st.subheader("Prediction Results (First 20 Rows)")
    st.dataframe(results.head(20))

    # Download option
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions (CSV)",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Summary and charts
    st.subheader("Summary Statistics")
    defect_rate = results["Predicted_Defect"].mean() * 100
    st.metric("Predicted Defect Rate", f"{defect_rate:.2f}%")

    counts = results["Predicted_Defect"].value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Top Predicted Defects (High Risk Modules)")
    st.dataframe(results.sort_values("Defect_Probability", ascending=False).head(10))

    # -------------------------------
    # Evaluation Mode
    # -------------------------------
    if eval_mode:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score,
            roc_curve, precision_recall_curve
        )
        import matplotlib.pyplot as plt

        st.subheader("Model Evaluation (using provided 'defect' labels)")

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
        st.write(f"**F1-Score:** {f1:.4f}")
        if roc_auc:
            st.write(f"**ROC-AUC:** {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, preds)
        st.write("Confusion Matrix (rows=True labels, cols=Predictions):")
        st.write(cm)

        # ROC & Precision-Recall Curves
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            pr_prec, pr_rec, _ = precision_recall_curve(y_true, probs)

            fig1, ax1 = plt.subplots()
            ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}" if roc_auc else "")
            ax1.plot([0,1],[0,1],'k--')
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
        except Exception as e:
            st.info("Unable to compute ROC/PR curves: " + str(e))

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
    - If included, choose whether to evaluate model accuracy or only predict.  
    - If column order differs, include `feature_order.txt` in this folder.  
    - This dashboard supports both local and Streamlit Cloud deployment.  
    """
)
