import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)

# PATHS

BASE_DIR = r"C:\5G_Network_Quality"

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "predictions"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evaluation"
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


LOGISTIC_PATH = os.path.join(
    PREDICTIONS_DIR,
    "logistic_handover_validation_predictions.csv"
)

XGBOOST_PATH = os.path.join(
    PREDICTIONS_DIR,
    "xgboost_handover_validation_predictions.csv"
)

# HEADER

print("MODEL EVALUATION")

print("""
This evaluation uses VALIDATION predictions only.

Test set is NOT loaded.

Models:
1. Logistic Regression baseline
2. XGBoost handover classifier

Evaluation:
- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve
- Threshold Analysis
""")

# LOAD PREDICTIONS

print("LOADING VALIDATION PREDICTIONS")

logistic = pd.read_csv(
    LOGISTIC_PATH
)

xgboost = pd.read_csv(
    XGBOOST_PATH
)

print("\nLogistic Regression:")
print(f"Rows    : {len(logistic):,}")
print(f"Columns : {len(logistic.columns):,}")

print("\nXGBoost:")
print(f"Rows    : {len(xgboost):,}")
print(f"Columns : {len(xgboost.columns):,}")

# VERIFY DATa

required_columns = [
    "actual_handover",
    "handover_probability"
]

for column in required_columns:

    if column not in logistic.columns:
        raise ValueError(
            f"Missing column in Logistic predictions: {column}"
        )

    if column not in xgboost.columns:
        raise ValueError(
            f"Missing column in XGBoost predictions: {column}"
        )

# PREPARE ARRAYS

y_true_logistic = logistic[
    "actual_handover"
].astype(int)

y_prob_logistic = logistic[
    "handover_probability"
].astype(float)

y_true_xgboost = xgboost[
    "actual_handover"
].astype(int)

y_prob_xgboost = xgboost[
    "handover_probability"
].astype(float)

# CHECK TARGET CONSISTENCY

if not np.array_equal(
    y_true_logistic.values,
    y_true_xgboost.values
):
    raise ValueError(
        "Logistic and XGBoost validation targets "
        "are not in the same order."
    )


y_true = y_true_logistic


print("\nTarget distribution:")
print(
    y_true.value_counts()
    .sort_index()
    .to_string()
)

# FUNCTION: METRICS AT THRESHOLD

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    }

# BASELINE METRICS AT 0.50

print("VALIDATION PERFORMANCE AT THRESHOLD = 0.50")

def print_model_metrics(
    model_name,
    y_true,
    probabilities
):

    threshold = 0.50

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(f"\n{model_name}")
    print("-" * 50)

    print(
        f"Precision : {metrics['precision']:.6f}"
    )

    print(
        f"Recall    : {metrics['recall']:.6f}"
    )

    print(
        f"F1-score  : {metrics['f1']:.6f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.6f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN = {metrics['true_negative']:,}"
    )

    print(
        f"FP = {metrics['false_positive']:,}"
    )

    print(
        f"FN = {metrics['false_negative']:,}"
    )

    print(
        f"TP = {metrics['true_positive']:,}"
    )

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"]
    }


logistic_summary = print_model_metrics(
    "Logistic Regression",
    y_true,
    y_prob_logistic
)

xgboost_summary = print_model_metrics(
    "XGBoost",
    y_true,
    y_prob_xgboost
)

# SAVE MODEL COMPARISON

model_comparison = pd.DataFrame([
    logistic_summary,
    xgboost_summary
])

comparison_path = os.path.join(
    EVALUATION_DIR,
    "handover_model_comparison.csv"
)

model_comparison.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nModel comparison saved to:\n{comparison_path}"
)

# THRESHOLD ANALYSIS

print("THRESHOLD ANALYSIS")

thresholds = np.round(
    np.arange(
        0.05,
        0.96,
        0.05
    ),
    2
)

threshold_results = []


for threshold in thresholds:

    logistic_metrics = calculate_metrics(
        y_true,
        y_prob_logistic,
        threshold
    )

    xgboost_metrics = calculate_metrics(
        y_true,
        y_prob_xgboost,
        threshold
    )

    threshold_results.append({
        "model": "Logistic Regression",
        **logistic_metrics
    })

    threshold_results.append({
        "model": "XGBoost",
        **xgboost_metrics
    })


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_path = os.path.join(
    EVALUATION_DIR,
    "handover_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)

print(
    f"\nThreshold analysis saved to:\n{threshold_path}"
)

# FIND BEST F1 THRESHOLD

print("BEST F1 THRESHOLD")

for model_name in [
    "Logistic Regression",
    "XGBoost"
]:

    model_thresholds = threshold_df[
        threshold_df["model"] == model_name
    ]

    best_row = model_thresholds.loc[
        model_thresholds["f1"].idxmax()
    ]

    print(f"\n{model_name}")

    print(
        f"Best threshold : "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Precision      : "
        f"{best_row['precision']:.6f}"
    )

    print(
        f"Recall         : "
        f"{best_row['recall']:.6f}"
    )

    print(
        f"F1-score       : "
        f"{best_row['f1']:.6f}"
    )

# CONFUSION MATRICES

print("CREATING CONFUSION MATRICES")

def save_confusion_matrix(
    model_name,
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Handover",
            "Handover"
        ]
    )

    display.plot(
        ax=ax
    )

    ax.set_title(
        f"{model_name} - Validation Confusion Matrix"
    )

    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_handover_confusion_matrix.png"
    )

    output_path = os.path.join(
        FIGURES_DIR,
        filename
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(output_path)


save_confusion_matrix(
    "Logistic Regression",
    y_true,
    y_prob_logistic
)

save_confusion_matrix(
    "XGBoost",
    y_true,
    y_prob_xgboost
)

# PRECISION-RECALL CURVE

print("\nCreating Precision-Recall curve...")


precision_logistic, recall_logistic, _ = (
    precision_recall_curve(
        y_true,
        y_prob_logistic
    )
)

precision_xgboost, recall_xgboost, _ = (
    precision_recall_curve(
        y_true,
        y_prob_xgboost
    )
)

pr_auc_logistic = average_precision_score(
    y_true,
    y_prob_logistic
)

pr_auc_xgboost = average_precision_score(
    y_true,
    y_prob_xgboost
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    recall_logistic,
    precision_logistic,
    label=f"Logistic Regression (AP={pr_auc_logistic:.3f})"
)

ax.plot(
    recall_xgboost,
    precision_xgboost,
    label=f"XGBoost (AP={pr_auc_xgboost:.3f})"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "Precision-Recall Curve - Handover Prediction"
)

ax.legend()
ax.grid(True)

fig.tight_layout()

pr_curve_path = os.path.join(
    FIGURES_DIR,
    "handover_pr_curve.png"
)

fig.savefig(
    pr_curve_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(pr_curve_path)

# ROC CURVE

print("\nCreating ROC curve...")


fpr_logistic, tpr_logistic, _ = (
    roc_curve(
        y_true,
        y_prob_logistic
    )
)

fpr_xgboost, tpr_xgboost, _ = (
    roc_curve(
        y_true,
        y_prob_xgboost
    )
)

roc_auc_logistic = roc_auc_score(
    y_true,
    y_prob_logistic
)

roc_auc_xgboost = roc_auc_score(
    y_true,
    y_prob_xgboost
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC={roc_auc_logistic:.3f})"
)

ax.plot(
    fpr_xgboost,
    tpr_xgboost,
    label=f"XGBoost (AUC={roc_auc_xgboost:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title(
    "ROC Curve - Handover Prediction"
)

ax.legend()
ax.grid(True)

fig.tight_layout()

roc_curve_path = os.path.join(
    FIGURES_DIR,
    "handover_roc_curve.png"
)

fig.savefig(
    roc_curve_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(roc_curve_path)

# THRESHOLD PLOT

print("\nCreating threshold analysis plot...")


fig, ax = plt.subplots(
    figsize=(9, 6)
)

for model_name in [
    "Logistic Regression",
    "XGBoost"
]:

    data = threshold_df[
        threshold_df["model"] == model_name
    ]

    ax.plot(
        data["threshold"],
        data["precision"],
        marker="o",
        label=f"{model_name} Precision"
    )

    ax.plot(
        data["threshold"],
        data["recall"],
        marker="s",
        label=f"{model_name} Recall"
    )

    ax.plot(
        data["threshold"],
        data["f1"],
        marker="^",
        label=f"{model_name} F1"
    )


ax.set_xlabel(
    "Probability Threshold"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Handover Model Performance vs Probability Threshold"
)

ax.legend()
ax.grid(True)

fig.tight_layout()

threshold_plot_path = os.path.join(
    FIGURES_DIR,
    "handover_threshold_analysis.png"
)

fig.savefig(
    threshold_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(threshold_plot_path)

# FINAL MESSAGE

print("""
Generated:

results/evaluation/
    handover_model_comparison.csv
    handover_threshold_analysis.csv

results/figures/
    logistic_regression_handover_confusion_matrix.png
    xgboost_handover_confusion_matrix.png
    handover_pr_curve.png
    handover_roc_curve.png
    handover_threshold_analysis.png

Test set was NOT used.
""")