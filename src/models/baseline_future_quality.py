import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("FUTURE QUALITY PERSISTENCE BASELINE")

print("""
Prediction strategy:
- Future RSRP = Current RSRP
- Future RSRQ = Current RSRQ
- Future SINR = Current SINR

Important:
- Test set is NOT loaded.
- Only rows with available future measurements are evaluated.
- No model is trained.
- No future information is used to create predictions.
""")


# PATHS

TRAIN_PATH = r"c:\5G_Network_Quality\data\train\features_train.csv"
VALIDATION_PATH = r"c:\5G_Network_Quality\data\validation\features_validation.csv"

RESULTS_DIR = r"c:\5G_Network_Quality\results"
OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_future_quality_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# LOAD DATA

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

# DEFINE CURRENT → FUTURE QUALITY PAIRS

quality_pairs = {
    "RSRP": ("serving_rsrp", "future_rsrp_raw"),
    "RSRQ": ("serving_rsrq", "future_rsrq_raw"),
    "SINR": ("serving_sinr", "future_sinr_raw"),
}

# EVALUATION FUNCTION

def evaluate_persistence(df, dataset_name):

    print("\n")
    print(f"{dataset_name.upper()} RESULTS")

    results = []

    for metric_name, (current_col, future_col) in quality_pairs.items():

        # Only evaluate rows where the future target exists.
        valid = df[
            df[current_col].notna()
            & df[future_col].notna()
        ].copy()

        y_true = valid[future_col].astype(float)
        y_pred = valid[current_col].astype(float)

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        r2 = r2_score(y_true, y_pred)

        print(f"\n{metric_name}")
        print("-" * 50)
        print(f"Current feature        : {current_col}")
        print(f"Future target          : {future_col}")
        print(f"Valid samples          : {len(valid):,}")
        print(f"MAE                    : {mae:.6f}")
        print(f"RMSE                   : {rmse:.6f}")
        print(f"R²                     : {r2:.6f}")

        results.append({
            "dataset": dataset_name,
            "metric": metric_name,
            "current_feature": current_col,
            "future_target": future_col,
            "valid_samples": len(valid),
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

    return results

# RUN BASELINE

all_results = []

all_results.extend(
    evaluate_persistence(train, "train")
)

all_results.extend(
    evaluate_persistence(validation, "validation")
)

# SAVE RESULTS

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# FINAL SUMMARY

print("\nResults:")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nTEST SET WAS NOT USED.")