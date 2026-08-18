# ML Assignment 2 — Breast Cancer Classification

## a. Problem Statement

The objective is to implement multiple machine learning classification models on one public classification dataset, evaluate their performance using standard classification metrics, and demonstrate the results through an interactive Streamlit web application.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset contains 569 instances and 30 numeric predictive features. The target is binary:

- **Benign**
- **Malignant**

The dataset was originally published through the UCI Machine Learning Repository. The scikit-learn copy used in this project corresponds to the same Breast Cancer Wisconsin (Diagnostic) dataset.

## c. GitHub Repository Link

**Replace this placeholder after creating the repository:**

`<GITHUB_REPOSITORY_LINK>`

## d. Models Used

The following classification models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

### Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9649 | 0.9970 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

### Observations on Model Performance

**Logistic Regression:** Performs well on this dataset because the standardized numeric features provide useful information for a linear decision boundary.

**Decision Tree:** Captures non-linear relationships and feature interactions, but its performance can depend strongly on tree depth and may be less stable than an ensemble.

**kNN:** Uses distances between standardized observations. Its performance depends on the neighborhood size and feature scaling.

**Naive Bayes:** Provides a fast probabilistic baseline. Its independence assumption can limit performance when features are correlated.

**Random Forest:** Combines many decision trees and can capture non-linear relationships while reducing the variance of an individual tree.

**Overall Winner:** Based on the highest test-set F1 score in this implementation, the current winner is **Logistic Regression**. This conclusion should be kept only after rerunning the supplied training/evaluation workflow and verifying the displayed results.

## Streamlit Application

The application provides:

- CSV test-data upload
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Comparison of all implemented models

## Live Streamlit App Link

**Replace this placeholder after deployment:**

`<STREAMLIT_APP_LINK>`

## Project Structure

```text
ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── train_models.py
└── model/
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    ├── random_forest.py
    └── *.joblib
```

## Reproducibility

The training workflow uses a stratified 80/20 train-test split with `random_state=42`. Run:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Dataset Source

UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic).
