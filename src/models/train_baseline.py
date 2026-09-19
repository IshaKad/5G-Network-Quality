import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE_DIR = r"c:\5G_Network_Quality"

TRAIN_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "train",
    "features_train.csv"
)

VALIDATION_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "validation",
    "features_validation.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "serving_cell",

    "serving_rsrp",
    "serving_rsrq",
    "serving_sinr",

    "serving_rsrp_present",
    "serving_rsrq_present",
    "serving_sinr_present",

    "neighbor_count",
    "has_neighbor",

    "neighbor_1_pci",
    "neighbor_1_rsrp",
    "neighbor_1_rsrq",
    "neighbor_1_sinr",

    "neighbor_1_rsrp_present",
    "neighbor_1_rsrq_present",
    "neighbor_1_sinr_present",

    "neighbor_1_rsrp_delta",
    "neighbor_1_rsrq_delta",
    "neighbor_1_sinr_delta",

    "neighbor_2_pci",
    "neighbor_2_rsrp",
    "neighbor_2_rsrq",
    "neighbor_2_sinr",

    "neighbor_2_rsrp_present",
    "neighbor_2_rsrq_present",
    "neighbor_2_sinr_present",

    "neighbor_2_rsrp_delta",
    "neighbor_2_rsrq_delta",
    "neighbor_2_sinr_delta",

    "neighbor_3_pci",
    "neighbor_3_rsrp",
    "neighbor_3_rsrq",
    "neighbor_3_sinr",

    "neighbor_3_rsrp_present",
    "neighbor_3_rsrq_present",
    "neighbor_3_sinr_present",

    "neighbor_3_rsrp_delta",
    "neighbor_3_rsrq_delta",
    "neighbor_3_sinr_delta",

    # BEST NEIGHBOR
    "best_neighbor_index",
    "best_neighbor_pci",
    "best_neighbor_rsrp",
    "best_neighbor_rsrp_delta",
    "best_neighbor_rsrq_delta",
    "best_neighbor_sinr_delta",

    # NEIGHBOR AGGREGATES
    "neighbor_rsrp_max",
    "neighbor_rsrp_min",
    "neighbor_rsrp_mean",
    "neighbor_rsrp_range",

    "neighbor_rsrq_max",
    "neighbor_rsrq_min",
    "neighbor_rsrq_mean",
    "neighbor_rsrq_range",

    "neighbor_sinr_max",
    "neighbor_sinr_min",
    "neighbor_sinr_mean",
    "neighbor_sinr_range",
]


TARGET_COLUMN = "handover_within_3s"

def load_dataset(path, dataset_name):
    print()
    print(f"LOADING {dataset_name}")

    df = pd.read_csv(path)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns in {dataset_name}: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in {dataset_name}."
        )

    return df

def prepare_data(df, dataset_name):
    X = df[FEATURE_COLUMNS].copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    if y.isna().any():
        raise ValueError(
            f"{dataset_name} contains invalid target values."
        )

    y = y.astype(int)

    print()
    print(f"{dataset_name} TARGET DISTRIBUTION")

    counts = y.value_counts().sort_index()

    for label, count in counts.items():
        percentage = (count / len(y)) * 100
        print(
            f"Class {label}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

    print()
    print(
        f"Missing feature values: "
        f"{X.isna().sum().sum():,}"
    )

    return X, y

# BUILD BASELINE PIPELINE

def build_model():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURE_COLUMNS
            )
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    return pipeline

def evaluate_model(model, X, y, dataset_name):
    probabilities = model.predict_proba(X)[:, 1]

    # Default probability threshold
    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print(f"{dataset_name} RESULTS")

    print(f"Threshold : {threshold:.2f}")
    print()
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print(f"PR-AUC    : {pr_auc:.6f}")
    print(f"ROC-AUC   : {roc_auc:.6f}")

    print()
    print("CONFUSION MATRIX")
    print("                Predicted")
    print("                0       1")
    print(
        f"Actual 0    {cm[0, 0]:7,d} {cm[0, 1]:7,d}"
    )
    print(
        f"Actual 1    {cm[1, 0]:7,d} {cm[1, 1]:7,d}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y,
            predictions,
            digits=6,
            zero_division=0
        )
    )

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }

print("BASELINE HANDOVER CLASSIFIER")

print()
print("Model:")
print("- Logistic Regression")
print("- Class-weight balanced")
print("- Median imputation")
print("- StandardScaler")
print()
print("Target:")
print("- handover_within_3s")
print()
print("Important:")
print("- Test set is NOT loaded.")
print("- Future columns are NOT used as features.")
print("- Handover ground-truth columns are NOT used as features.")
print("- Preprocessing is fitted only on training data.")
print()

train_df = load_dataset(
    TRAIN_INPUT,
    "TRAIN"
)

validation_df = load_dataset(
    VALIDATION_INPUT,
    "VALIDATION"
)

X_train, y_train = prepare_data(
    train_df,
    "TRAIN"
)

X_validation, y_validation = prepare_data(
    validation_df,
    "VALIDATION"
)

print()
print("BUILDING BASELINE MODEL")

model = build_model()

# TRAIN

print()
print("Training Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# EVALUATE TRAIN

train_results = evaluate_model(
    model,
    X_train,
    y_train,
    "TRAIN"
)

# EVALUATE VALIDATION

validation_results = evaluate_model(
    model,
    X_validation,
    y_validation,
    "VALIDATION"
)

results = pd.DataFrame(
    [
        train_results,
        validation_results
    ]
)

results_path = os.path.join(
    RESULTS_DIR,
    "baseline_handover_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print()
print("COMPLETE")

print()
print("Results saved to:")
print(results_path)

print()
print("TEST SET WAS NOT USED.")
print()