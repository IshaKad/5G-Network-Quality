import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBRegressor, XGBClassifier

print("XGBOOST MODELS")

print("""
Models:
1. XGBoost Future RSRP Regression
2. XGBoost Future RSRQ Regression
3. XGBoost Future SINR Regression
4. XGBoost Handover-within-3s Classification

Important:
- Test set is NOT loaded.
- Future columns are NOT used as input features.
- Handover ground-truth columns are NOT used as input features.
- Training uses TRAIN only.
- Validation is used for model monitoring.
- PCI/cell identifiers are treated as categorical variables.
""")

TRAIN_PATH = r"c:\5G_Network_Quality\data\train\features_train.csv"
VALIDATION_PATH = r"c:\5G_Network_Quality\data\validation\features_validation.csv"

RESULTS_DIR = r"c:\5G_Network_Quality\results"

REGRESSION_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "xgboost_future_quality_results.csv"
)

CLASSIFICATION_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "xgboost_handover_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n")
print("LOADING TRAIN")

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

print(f"Rows    : {len(train):,}")
print(f"Columns : {len(train.columns):,}")


print("\n")
print("LOADING VALIDATION")

validation = pd.read_csv(
    VALIDATION_PATH,
    low_memory=False
)

print(f"Rows    : {len(validation):,}")
print(f"Columns : {len(validation.columns):,}")

# Categorical identifiers.
categorical_features = [
    "serving_cell",
    "neighbor_1_pci",
    "neighbor_2_pci",
    "neighbor_3_pci",
    "best_neighbor_pci",
]


# Numerical current-state features.
numeric_features = [
    "serving_rsrp",
    "serving_rsrq",
    "serving_sinr",

    "serving_rsrp_present",
    "serving_rsrq_present",
    "serving_sinr_present",

    "neighbor_count",
    "has_neighbor",

    "neighbor_1_rsrp",
    "neighbor_1_rsrq",
    "neighbor_1_sinr",
    "neighbor_1_rsrp_present",
    "neighbor_1_rsrq_present",
    "neighbor_1_sinr_present",
    "neighbor_1_rsrp_delta",
    "neighbor_1_rsrq_delta",
    "neighbor_1_sinr_delta",

    "neighbor_2_rsrp",
    "neighbor_2_rsrq",
    "neighbor_2_sinr",
    "neighbor_2_rsrp_present",
    "neighbor_2_rsrq_present",
    "neighbor_2_sinr_present",
    "neighbor_2_rsrp_delta",
    "neighbor_2_rsrq_delta",
    "neighbor_2_sinr_delta",

    "neighbor_3_rsrp",
    "neighbor_3_rsrq",
    "neighbor_3_sinr",
    "neighbor_3_rsrp_present",
    "neighbor_3_rsrq_present",
    "neighbor_3_sinr_present",
    "neighbor_3_rsrp_delta",
    "neighbor_3_rsrq_delta",
    "neighbor_3_sinr_delta",

    "best_neighbor_index",
    "best_neighbor_rsrp",
    "best_neighbor_rsrp_delta",
    "best_neighbor_rsrq_delta",
    "best_neighbor_sinr_delta",

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


feature_columns = categorical_features + numeric_features

# VERIFY FEATURES

missing_train_features = [
    col for col in feature_columns
    if col not in train.columns
]

missing_validation_features = [
    col for col in feature_columns
    if col not in validation.columns
]

if missing_train_features:
    raise ValueError(
        f"Missing train features: {missing_train_features}"
    )

if missing_validation_features:
    raise ValueError(
        f"Missing validation features: {missing_validation_features}"
    )

print("\n")
print("PREPARING FEATURES")

for col in categorical_features:

    # Convert identifiers to strings first.
    train[col] = train[col].fillna("MISSING").astype(str)
    validation[col] = validation[col].fillna("MISSING").astype(str)

    # Use the same category definitions in train and validation.
    categories = sorted(
        set(train[col].unique())
        | set(validation[col].unique())
    )

    train[col] = pd.Categorical(
        train[col],
        categories=categories
    )

    validation[col] = pd.Categorical(
        validation[col],
        categories=categories
    )

X_train = train[feature_columns].copy()
X_validation = validation[feature_columns].copy()

print(f"Total input features : {len(feature_columns)}")
print(f"Categorical features  : {len(categorical_features)}")
print(f"Numeric features      : {len(numeric_features)}")

common_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}

print("\n")
print("FUTURE QUALITY REGRESSION")


quality_targets = {
    "RSRP": "future_rsrp_raw",
    "RSRQ": "future_rsrq_raw",
    "SINR": "future_sinr_raw",
}


regression_results = []


for metric_name, target_column in quality_targets.items():

    print("\n")
    print(f"TRAINING XGBOOST FOR FUTURE {metric_name}")

    # Only rows with a known future target are usable.
    train_mask = train[target_column].notna()
    validation_mask = validation[target_column].notna()

    Xtr = X_train.loc[train_mask]
    ytr = train.loc[train_mask, target_column].astype(float)

    Xval = X_validation.loc[validation_mask]
    yval = validation.loc[
        validation_mask,
        target_column
    ].astype(float)

    print(f"Training samples   : {len(Xtr):,}")
    print(f"Validation samples : {len(Xval):,}")

    model = XGBRegressor(
        objective="reg:squarederror",
        **common_params
    )

    print("Training...")

    model.fit(
        Xtr,
        ytr,
        eval_set=[
            (Xtr, ytr),
            (Xval, yval)
        ],
        verbose=False
    )

    print("Training complete.")

    predictions = model.predict(Xval)

    mae = mean_absolute_error(
        yval,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            yval,
            predictions
        )
    )

    r2 = r2_score(
        yval,
        predictions
    )

    print(f"\n{metric_name} VALIDATION RESULTS")
    print("-" * 50)
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    regression_results.append({
        "dataset": "validation",
        "metric": metric_name,
        "valid_samples": len(Xval),
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })

regression_results_df = pd.DataFrame(
    regression_results
)

regression_results_df.to_csv(
    REGRESSION_RESULTS_PATH,
    index=False
)

print("HANDOVER CLASSIFICATION")


target_column = "handover_within_3s"

y_train = train[target_column].astype(int)
y_validation = validation[target_column].astype(int)


print("\nTRAIN TARGET DISTRIBUTION")

print(
    y_train.value_counts()
    .sort_index()
    .to_string()
)

print("\nVALIDATION TARGET DISTRIBUTION")

print(
    y_validation.value_counts()
    .sort_index()
    .to_string()
)

# CLASS IMBALANCE

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count / positive_count
)

print("\nClass imbalance handling:")
print(
    f"scale_pos_weight = "
    f"{scale_pos_weight:.6f}"
)

# BUILD CLASSIFIER

classifier = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,

    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,

    tree_method="hist",
    enable_categorical=True,

    random_state=42,
    n_jobs=-1
)


print("\nTraining XGBoost handover classifier...")

classifier.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation)
    ],
    verbose=False
)

print("Training complete.")

validation_probabilities = classifier.predict_proba(
    X_validation
)[:, 1]

threshold = 0.50

validation_predictions = (
    validation_probabilities >= threshold
).astype(int)

precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

pr_auc = average_precision_score(
    y_validation,
    validation_probabilities
)

roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_validation,
    validation_predictions
).ravel()


print("\n")
print("XGBOOST HANDOVER VALIDATION RESULTS")

print(f"Threshold : {threshold:.2f}")

print(f"\nPrecision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"PR-AUC    : {pr_auc:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")

print("\nCONFUSION MATRIX")
print("-" * 50)

print("                Predicted")
print("                0       1")
print(
    f"Actual 0    {tn:6,} {fp:7,}"
)
print(
    f"Actual 1    {fn:6,} {tp:7,}"
)

# SAVE CLASSIFICATION RESULTS

classification_results = pd.DataFrame([
    {
        "dataset": "validation",
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "scale_pos_weight": scale_pos_weight
    }
])

classification_results.to_csv(
    CLASSIFICATION_RESULTS_PATH,
    index=False
)

print("\n")

print("\nFuture-quality results saved to:")
print(REGRESSION_RESULTS_PATH)

print("\nHandover-classification results saved to:")
print(CLASSIFICATION_RESULTS_PATH)

print("\nTEST SET WAS NOT USED.")