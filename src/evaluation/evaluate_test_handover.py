import os
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = r"C:\5G_Network_Quality"

TEST_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test",
    "features_test.csv"
)

ENGINE_RESULT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "handover",
    "intelligent_handover_test_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading test ground truth...")

test_data = pd.read_csv(TEST_FEATURE_PATH)

print(f"Test samples: {len(test_data):,}")

print("Loading intelligent handover results...")

results = pd.read_csv(ENGINE_RESULT_PATH)

if len(test_data) != len(results):
    raise ValueError(
        "Test data and engine results have different row counts."
    )

y_true = test_data["handover_within_3s"].astype(int)

y_pred = (
    results["decision"]
    .eq("HANDOVER")
    .astype(int)
)

handover_probability = results[
    "handover_probability"
].astype(float)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_true,
    handover_probability
)

roc_auc = roc_auc_score(
    y_true,
    handover_probability
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

print()
print("Final Test Results")
print()

print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")

print()

print("Confusion Matrix")
print()

print(f"TN: {tn:,}")
print(f"FP: {fp:,}")
print(f"FN: {fn:,}")
print(f"TP: {tp:,}")

print()

print("Test class distribution")

print(
    y_true.value_counts().sort_index()
)

print()

print("Predicted decision distribution")

print(
    results["decision"].value_counts()
)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "NO_HANDOVER",
        "HANDOVER"
    ],
    zero_division=0
)

print()
print(report)

metrics = pd.DataFrame({
    "metric": [
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives"
    ],
    "value": [
        precision,
        recall,
        f1,
        pr_auc,
        roc_auc,
        tn,
        fp,
        fn,
        tp
    ]
})

output_path = os.path.join(
    OUTPUT_DIR,
    "intelligent_handover_test_metrics.csv"
)

metrics.to_csv(
    output_path,
    index=False
)

print(
    f"Metrics saved to: {output_path}"
)