import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="ML Assignment 2", page_icon="🧠", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

FEATURE_COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }

def calculate_metrics(model, X, y):
    pred = model.predict(X)
    classes = list(model.classes_)
    malignant_index = classes.index("Malignant")
    malignant_probability = model.predict_proba(X)[:, malignant_index]

    y_binary = (y == "Malignant").astype(int)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y_binary, malignant_probability),
        "Precision": precision_score(y, pred, pos_label="Malignant", zero_division=0),
        "Recall": recall_score(y, pred, pos_label="Malignant", zero_division=0),
        "F1": f1_score(y, pred, pos_label="Malignant", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred

st.title("🧠 Breast Cancer Classification — ML Assignment 2")
st.caption("BITS WILP M.Tech AIML/DSE | Interactive model evaluation")

st.markdown(
    """
    **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
    **Target:** Benign / Malignant  
    **Models:** Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
    """
)

models = load_models()

uploaded = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Upload the supplied test_data.csv file. It must contain the 30 feature columns and a target column."
)

default_path = ROOT / "test_data.csv"
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_name = uploaded.name
elif default_path.exists():
    df = pd.read_csv(default_path)
    source_name = "Bundled test_data.csv"
else:
    df = None
    source_name = None

if df is None:
    st.info("Upload test_data.csv to begin.")
    st.stop()

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if "target" not in df.columns:
    st.error("The CSV must contain a 'target' column for evaluation.")
    st.stop()
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X_test = df[FEATURE_COLUMNS]
y_test = df["target"].astype(str)

if not set(y_test.unique()).issubset({"Benign", "Malignant"}):
    st.error("The target column must contain only 'Benign' and 'Malignant'.")
    st.stop()

st.success(f"Loaded {len(df)} test records from {source_name}.")

selected_model = st.selectbox("Select a model", list(models.keys()))
model = models[selected_model]

metrics, predictions = calculate_metrics(model, X_test, y_test)

st.subheader(f"{selected_model} — Evaluation Metrics")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    labels = ["Benign", "Malignant"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

st.subheader("All Model Results")
comparison = []
for name, fitted_model in models.items():
    model_metrics, _ = calculate_metrics(fitted_model, X_test, y_test)
    comparison.append({"ML Model Name": name, **model_metrics})

comparison_df = pd.DataFrame(comparison)
st.dataframe(comparison_df.round(4), use_container_width=True)

winner = comparison_df.loc[comparison_df["F1"].idxmax(), "ML Model Name"]
st.info(f"Overall winner by test-set F1 score: **{winner}**")
