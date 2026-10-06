import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = r"C:\5G_Network_Quality"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_handover_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "validation",
    "features_validation.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "handover"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

HANDOVER_THRESHOLD = 0.90
RSRP_MARGIN = 2

print("Loading validation features...")

data = pd.read_csv(FEATURE_PATH)

print(f"Validation samples: {len(data):,}")


print("Loading XGBoost handover model...")

model = joblib.load(MODEL_PATH)


model_features = list(
    model.get_booster().feature_names
)

print(f"Model expects {len(model_features)} features.")


missing_features = [
    column
    for column in model_features
    if column not in data.columns
]

if missing_features:

    raise ValueError(
        "Missing features required by the model: "
        + ", ".join(missing_features)
    )


extra_features = [
    column
    for column in data.columns
    if column not in model_features
]

print(f"Extra non-model columns: {len(extra_features)}")


X = data[model_features].copy()


categorical_features = [
    "serving_cell",
    "neighbor_1_pci",
    "neighbor_2_pci",
    "neighbor_3_pci",
    "best_neighbor_pci"
]

TRAIN_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train",
    "features_train.csv"
)

print("Loading training feature categories...")

train_data = pd.read_csv(
    TRAIN_FEATURE_PATH,
    usecols=categorical_features
)

for column in categorical_features:

    categories = (
        train_data[column]
        .dropna()
        .astype(str)
        .unique()
    )

    values = X[column].copy()

    X[column] = pd.Categorical(
        values.astype("string"),
        categories=categories
    )

print("Model input features are correctly ordered.")

print("Generating handover probabilities...")

probabilities = model.predict_proba(X)[:, 1]


def numeric_value(value):

    try:
        return float(value)

    except (TypeError, ValueError):

        return np.nan


def find_best_neighbor(row):

    serving_rsrp = numeric_value(
        row["serving_rsrp"]
    )

    if pd.isna(serving_rsrp):

        return None


    neighbors = []


    for i in range(1, 4):

        pci = row[f"neighbor_{i}_pci"]

        rsrp = numeric_value(
            row[f"neighbor_{i}_rsrp"]
        )

        if pd.isna(pci) or pd.isna(rsrp):

            continue


        neighbors.append({
            "neighbor_index": i,
            "pci": pci,
            "rsrp": rsrp
        })


    if not neighbors:

        return None


    neighbors.sort(
        key=lambda x: x["rsrp"],
        reverse=True
    )


    best = neighbors[0]


    improvement = (
        best["rsrp"] - serving_rsrp
    )


    if improvement < RSRP_MARGIN:

        return None


    best["improvement"] = improvement

    return best


def make_decision(row, probability):

    serving_cell = row["serving_cell"]


    if probability < HANDOVER_THRESHOLD:

        return {
            "decision": "NO_HANDOVER",
            "serving_cell": serving_cell,
            "target_pci": None,
            "target_neighbor_index": None,
            "handover_probability": probability,
            "neighbor_rsrp": None,
            "rsrp_improvement": None,
            "reason": (
                "Handover probability is below "
                "the decision threshold."
            )
        }


    best_neighbor = find_best_neighbor(row)


    if best_neighbor is None:

        return {
            "decision": "NO_HANDOVER",
            "serving_cell": serving_cell,
            "target_pci": None,
            "target_neighbor_index": None,
            "handover_probability": probability,
            "neighbor_rsrp": None,
            "rsrp_improvement": None,
            "reason": (
                "No neighbor provides sufficient "
                "RSRP improvement."
            )
        }


    return {
        "decision": "HANDOVER",
        "serving_cell": serving_cell,
        "target_pci": best_neighbor["pci"],
        "target_neighbor_index": (
            best_neighbor["neighbor_index"]
        ),
        "handover_probability": probability,
        "neighbor_rsrp": best_neighbor["rsrp"],
        "rsrp_improvement": best_neighbor["improvement"],
        "reason": (
            "Best neighbor provides sufficient "
            "RSRP improvement."
        )
    }


print("Making handover decisions...")

results = []


for index, row in data.iterrows():

    probability = float(
        probabilities[index]
    )


    decision = make_decision(
        row,
        probability
    )


    decision["measurement_id"] = row[
        "measurement_id"
    ]

    decision["ue_id"] = row[
        "ue_id"
    ]

    decision["measurement_time"] = row[
        "measurement_time"
    ]


    results.append(decision)


results_df = pd.DataFrame(results)


output_path = os.path.join(
    OUTPUT_DIR,
    "intelligent_handover_validation_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print()
print("Intelligent handover engine completed.")

print(
    f"Results saved to: {output_path}"
)

print()

print("Decision distribution:")

print(
    results_df["decision"].value_counts()
)

print()

print(
    f"Handover threshold: "
    f"{HANDOVER_THRESHOLD}"
)

print(
    f"Required RSRP improvement: "
    f"{RSRP_MARGIN}"
)