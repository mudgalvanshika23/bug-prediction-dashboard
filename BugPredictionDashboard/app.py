# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Bug Prediction Dashboard", layout="wide")

# --- Helper functions ---
def load_artifact(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path} — please place it in the app folder.")
        st.stop()
    return joblib.load(path)

def preprocess_input(df, imputer, scaler, expected_n_features=None):
    # Keep numeric columns only (coerce)
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # If we have fewer/extra cols than expected, handle
    if expected_n_features is not None and X.shape[1] != expected_n_features:
        st.warning(
            f"Uploaded data has {X.shape[1]} features but the model expects {expected_n_features}. "
            "If column order differs from training, provide columns in the same order or provide a file named 'feature_order.txt'."
        )
        # Try feature_order.txt to reorder
        if os.path.exists("feature_order.txt"):
            with open("feature_order.txt", "r") as fh:
                order = [l.strip() for l in fh.readlines() if l.strip()]
            # keep only the features that are present in both (and in the correct order)
            matched = [c for c in order if c in X.columns]
            if len(matched) == expected_n_features:
                X = X[matched]
                st.info("Reordered input using feature_order.txt")
            else:
                st.error("feature_order.txt doesn't match uploaded file columns. Unable to proceed.")
                st.stop()
        else:
            st.error("Feature count mismatch and no feature_order.txt found. Please upload correct file.")
            st.stop()

    # Imputation and scaling
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled, X

# --- Load models and preprocessors ---
st.sidebar.title("Model artifacts")
st.sidebar.write("Ensure these files are in the app folder:")
st.sidebar.write("- bug_predict_rf_tuned.joblib (required)")
st.sidebar.write("- imputer.joblib (required)")
st.sidebar.write("- scaler.joblib (required)")
st.sidebar.write("- (optional) feature_order.txt to map input columns")

try:
    model = load_artifact("bug_predict_rf_tuned.joblib")
    imputer = load_artifact("imputer.joblib")
    scaler = load_artifact("scaler.joblib")
except Exception as e:
    st.stop()

# expected feature count (if available from scaler)
expected_n_features = None
try:
    expected_n_features = getattr(scaler, "n_features_in_", None)
except Exception:
    expected_n_features = None

# App title and description
st.title("Software Defect Prediction Dashboard")
st.markdown(
    """
    Upload a CSV file containing software metrics (same columns used during training).
    The dashboard will preprocess the data using the saved imputer/scaler, predict defect probability
    using the tuned Random Forest model, and present results and visualizations.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload software metrics CSV file", type=["csv"])

# Sample file download / example
with st.expander("Sample dataset and guidance"):
    st.write("Use a JM1-like CSV with numeric columns and a consistent column order.")
    if st.button("Download sample jm1.csv (provided)"):
        if os.path.exists("jm1.csv"):
            with open("jm1.csv", "rb") as fh:
                st.download_button("Download sample jm1.csv", fh, file_name="jm1_sample.csv")
        else:
            st.info("No jm1.csv found in the folder.")

# Run prediction pipeline if file uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Unable to read CSV. Please ensure it's comma-separated and well-formed.")
        st.stop()

    st.subheader("Uploaded data preview")
    st.dataframe(df.head())

    # Basic checks
    if df.shape[0] == 0:
        st.error("Uploaded file contains no rows.")
        st.stop()

    # Preprocess
    X_scaled, X_original = preprocess_input(df, imputer, scaler, expected_n_features=expected_n_features)

    # Predictions
    preds = model.predict(X_scaled)
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        # If model doesn't support predict_proba fallback to confidences
        probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))

    # Prepare results DF
    results = X_original.copy()
    results["Predicted_Defect"] = preds
    results["Defect_Probability"] = np.round(probs, 4)

    st.subheader("Prediction results (first 20 rows)")
    st.dataframe(results.head(20))

    # Download results (CSV)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

    # Summary metrics
    st.subheader("Summary")
    defect_rate = results["Predicted_Defect"].mean() * 100
    st.metric("Predicted Defect Rate", f"{defect_rate:.2f}%")

    # Counts and chart
    counts = results["Predicted_Defect"].value_counts().sort_index()
    st.write("Predicted class counts (0 = non-defective, 1 = defective)")
    st.bar_chart(counts)

    # Show top-k risky modules
    st.subheader("Top predicted defect probabilities")
    topk = results.sort_values("Defect_Probability", ascending=False).head(10)
    st.dataframe(topk)

    # Optional: show feature influence placeholder
    st.info("If SHAP outputs are available offline, you can integrate them here for explainability.")

else:
    st.info("Upload a software metrics CSV to begin prediction.")

# Footer
st.markdown("---")
st.markdown("**Notes:**\n- The uploaded CSV must use the same features (and ideally the same order) as used during training.\n- If your column order differs, add a file named `feature_order.txt` with one feature name per line representing the training order.\n- For deployment to Streamlit sharing or other hosting, include all `.joblib` files and this `app.py` in a GitHub repo.")
