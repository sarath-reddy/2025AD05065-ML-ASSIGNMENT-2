from pathlib import Path
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target.map({0: "Malignant", 1: "Benign"})

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
}

for filename, model in models.items():
    model.fit(train_X, train_y)
    joblib.dump(model, MODEL_DIR / f"{filename}.joblib")

test_data = test_X.copy()
test_data["target"] = test_y.values
test_data.to_csv(ROOT / "test_data.csv", index=False)
print("Training complete. Test data and model artifacts were saved.")
