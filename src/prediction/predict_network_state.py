import os
import joblib
import numpy as np
import pandas as pd
from xgboost import data


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


HANDOVER_THRESHOLD = 0.90
RSRP_MARGIN = 2


def load_models():
    """Load all trained models and feature metadata."""

    models = {
        "rsrp": joblib.load(
            os.path.join(
                MODEL_DIR,
                "xgboost_future_rsrp_model.pkl"
            )
        ),
        "rsrq": joblib.load(
            os.path.join(
                MODEL_DIR,
                "xgboost_future_rsrq_model.pkl"
            )
        ),
        "sinr": joblib.load(
            os.path.join(
                MODEL_DIR,
                "xgboost_future_sinr_model.pkl"
            )
        ),
        "handover": joblib.load(
            os.path.join(
                MODEL_DIR,
                "xgboost_handover_model.pkl"
            )
        ),
        "metadata": joblib.load(
            os.path.join(
                MODEL_DIR,
                "xgboost_feature_metadata.pkl"
            )
        )
    }

    return models


def prepare_features(data, metadata):
    """Prepare one or more rows using the same feature structure as training."""

    data = data.copy()

    feature_columns = metadata["feature_columns"]
    categorical_features = metadata["categorical_features"]
    categorical_categories = metadata["categorical_categories"]

    for col in categorical_features:

        if col not in data.columns:
            data[col] = np.nan

        values = data[col].copy()

        values = values.where(
            values.notna(),
            np.nan
        )

        known_categories = categorical_categories[col]

        values = values.where(
            values.isin(known_categories),
            np.nan
        )

        data[col] = pd.Categorical(
            values,
            categories=known_categories
        )

    missing_features = [
        col
        for col in feature_columns
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    return data[feature_columns].copy()


def predict_network_state(data):
    """
    Predict future network quality and intelligent handover decision.

    Parameters
    ----------
    data : pandas.DataFrame
        Current-state feature rows.

    Returns
    -------
    pandas.DataFrame
        Prediction results.
    """

    models = load_models()

    metadata = models["metadata"]

    X = prepare_features(
        data,
        metadata
    )

    rsrp_prediction = models["rsrp"].predict(X)
    rsrq_prediction = models["rsrq"].predict(X)
    sinr_prediction = models["sinr"].predict(X)

    handover_probability = (
        models["handover"]
        .predict_proba(X)[:, 1]
    )

    results = data.copy()

    results["predicted_rsrp"] = rsrp_prediction
    results["predicted_rsrq"] = rsrq_prediction
    results["predicted_sinr"] = sinr_prediction

    results["handover_probability"] = (
        handover_probability
    )

    decisions = []
    target_pcis = []
    reasons = []

    for index, row in results.iterrows():

        probability = row["handover_probability"]

        current_rsrp = row["serving_rsrp"]

        best_pci = None
        best_rsrp = None

        neighbors = []

        for i in range(1, 4):

            pci_column = f"neighbor_{i}_pci"
            rsrp_column = f"neighbor_{i}_rsrp"
            present_column = f"neighbor_{i}_rsrp_present"

            if (
                pci_column in row.index
                and rsrp_column in row.index
            ):

                pci = row[pci_column]
                rsrp = row[rsrp_column]

                if pd.isna(pci) or pd.isna(rsrp):
                    continue

                if present_column in row.index:

                    present = row[present_column]

                    if not pd.isna(present):
                        if int(present) == 0:
                            continue

                neighbors.append(
                    {
                        "pci": pci,
                        "rsrp": float(rsrp)
                    }
                )

        if neighbors:

            best_neighbor = max(
                neighbors,
                key=lambda x: x["rsrp"]
            )

            best_pci = best_neighbor["pci"]
            best_rsrp = best_neighbor["rsrp"]

        if probability < HANDOVER_THRESHOLD:

            decisions.append("NO_HANDOVER")
            target_pcis.append(None)

            reasons.append(
                "The connection is currently stable "
                "and the predicted handover probability "
                "is below the decision threshold."
            )

            continue

        if best_rsrp is None:

            decisions.append("NO_HANDOVER")
            target_pcis.append(None)

            reasons.append(
                "A handover may be needed, but no suitable "
                "neighboring cell is available."
            )

            continue

        improvement = (
            best_rsrp - current_rsrp
        )

        if improvement < RSRP_MARGIN:

            decisions.append("NO_HANDOVER")
            target_pcis.append(None)

            reasons.append(
                "A handover is possible, but no neighboring "
                "cell provides enough signal improvement."
            )

            continue

        decisions.append("HANDOVER")
        target_pcis.append(best_pci)

        reasons.append(
            "The model predicts a likely handover and "
            "a neighboring cell provides stronger signal."
        )

    results["decision"] = decisions
    results["recommended_target_pci"] = target_pcis
    results["decision_reason"] = reasons

    return results


if __name__ == "__main__":

    print("Testing network prediction layer...")

    test_path = os.path.join(
        BASE_DIR,
        "data",
        "validation",
        "features_validation.csv"
    )

    validation = pd.read_csv(
        test_path,
        low_memory=False
    )

    sample = validation.head(10)

    predictions = predict_network_state(
        sample
    )

    output_columns = [
        "serving_cell",
        "serving_rsrp",
        "serving_rsrq",
        "serving_sinr",
        "predicted_rsrp",
        "predicted_rsrq",
        "predicted_sinr",
        "handover_probability",
        "decision",
        "recommended_target_pci",
        "decision_reason"
    ]

    print(
        predictions[output_columns].to_string(
            index=False
        )
    )

    print("\nPrediction layer test completed.")