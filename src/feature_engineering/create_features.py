import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = r"c:\5G_Network_Quality"

TRAIN_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "train",
    "ml_train.csv"
)

VALIDATION_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "validation",
    "ml_validation.csv"
)

TEST_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "test",
    "ml_test.csv"
)

TRAIN_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "train",
    "features_train.csv"
)

VALIDATION_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "validation",
    "features_validation.csv"
)

TEST_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "test",
    "features_test.csv"
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def to_float(value):
    """
    Convert a CSV value to float.

    Empty strings are treated as missing values.
    """

    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return float(value)

    except ValueError:
        return None


def to_int(value):
    """
    Convert a CSV value to integer.

    Handles:
    - True / False
    - 1 / 0
    - numeric strings
    - empty values

    Empty or unrecognized values are treated as missing.
    """
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    # Handle Boolean values stored as strings
    if value.lower() == "true":
        return 1

    if value.lower() == "false":
        return 0

    # Handle numeric values
    try:
        return int(float(value))
    except ValueError:
        return None


def format_number(value):
    """
    Store numeric feature values cleanly in CSV.

    Integers remain integers.
    Floating-point values retain useful precision.
    """

    if value is None:
        return ""

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):

        if value.is_integer():
            return str(int(value))

        return f"{value:.6f}"

    return str(value)


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def engineer_row(row):
    """
    Create engineered features for one measurement row.

    Important:
    - Only current measurement information is used.
    - Future columns are NOT used.
    - Handover ground-truth columns are NOT used as features.
    """

    features = {}

    # ------------------------------------------------------
    # SERVING CELL FEATURES
    # ------------------------------------------------------

    serving_rsrp = to_float(
        row["serving_rsrp_raw"]
    )

    serving_rsrq = to_float(
        row["serving_rsrq_raw"]
    )

    serving_sinr = to_float(
        row["serving_sinr_raw"]
    )

    serving_rsrp_present = to_int(
        row["serving_rsrp_present"]
    )

    serving_rsrq_present = to_int(
        row["serving_rsrq_present"]
    )

    serving_sinr_present = to_int(
        row["serving_sinr_present"]
    )


    features["serving_rsrp"] = serving_rsrp
    features["serving_rsrq"] = serving_rsrq
    features["serving_sinr"] = serving_sinr

    features["serving_rsrp_present"] = (
        serving_rsrp_present
    )

    features["serving_rsrq_present"] = (
        serving_rsrq_present
    )

    features["serving_sinr_present"] = (
        serving_sinr_present
    )


    # ------------------------------------------------------
    # NEIGHBOR EXTRACTION
    # ------------------------------------------------------

    neighbors = []

    for i in range(1, 4):

        pci = to_int(
            row[f"neighbor_{i}_pci"]
        )

        rsrp = to_float(
            row[f"neighbor_{i}_rsrp_raw"]
        )

        rsrq = to_float(
            row[f"neighbor_{i}_rsrq_raw"]
        )

        sinr = to_float(
            row[f"neighbor_{i}_sinr_raw"]
        )

        rsrp_present = to_int(
            row[f"neighbor_{i}_rsrp_present"]
        )

        rsrq_present = to_int(
            row[f"neighbor_{i}_rsrq_present"]
        )

        sinr_present = to_int(
            row[f"neighbor_{i}_sinr_present"]
        )


        # --------------------------------------------------
        # RETAIN INDIVIDUAL NEIGHBOR FEATURES
        # --------------------------------------------------

        features[
            f"neighbor_{i}_pci"
        ] = pci

        features[
            f"neighbor_{i}_rsrp"
        ] = rsrp

        features[
            f"neighbor_{i}_rsrq"
        ] = rsrq

        features[
            f"neighbor_{i}_sinr"
        ] = sinr

        features[
            f"neighbor_{i}_rsrp_present"
        ] = rsrp_present

        features[
            f"neighbor_{i}_rsrq_present"
        ] = rsrq_present

        features[
            f"neighbor_{i}_sinr_present"
        ] = sinr_present


        # --------------------------------------------------
        # SERVING → NEIGHBOR DELTAS
        # --------------------------------------------------

        if (
            serving_rsrp is not None
            and rsrp is not None
        ):

            features[
                f"neighbor_{i}_rsrp_delta"
            ] = rsrp - serving_rsrp

        else:

            features[
                f"neighbor_{i}_rsrp_delta"
            ] = None


        if (
            serving_rsrq is not None
            and rsrq is not None
        ):

            features[
                f"neighbor_{i}_rsrq_delta"
            ] = rsrq - serving_rsrq

        else:

            features[
                f"neighbor_{i}_rsrq_delta"
            ] = None


        if (
            serving_sinr is not None
            and sinr is not None
        ):

            features[
                f"neighbor_{i}_sinr_delta"
            ] = sinr - serving_sinr

        else:

            features[
                f"neighbor_{i}_sinr_delta"
            ] = None


        # --------------------------------------------------
        # STORE VALID NEIGHBOR
        # --------------------------------------------------

        if (
            pci is not None
            or rsrp is not None
            or rsrq is not None
            or sinr is not None
        ):

            neighbors.append(
                {
                    "index": i,
                    "pci": pci,
                    "rsrp": rsrp,
                    "rsrq": rsrq,
                    "sinr": sinr
                }
            )


    # ======================================================
    # NEIGHBOR COUNT
    # ======================================================

    neighbor_count = len(neighbors)

    features["neighbor_count"] = neighbor_count

    features["has_neighbor"] = (
        1 if neighbor_count > 0 else 0
    )


    # ======================================================
    # BEST NEIGHBOR
    # ======================================================

    # We use the highest available raw RSRP value.
    #
    # IMPORTANT:
    # These are encoded values, so we call this
    # "highest raw RSRP" rather than interpreting it
    # as a physical dBm measurement.

    valid_rsrp_neighbors = [
        n
        for n in neighbors
        if n["rsrp"] is not None
    ]


    if valid_rsrp_neighbors:

        best_neighbor = max(
            valid_rsrp_neighbors,
            key=lambda n: n["rsrp"]
        )

        best_neighbor_index = (
            best_neighbor["index"]
        )

        best_neighbor_pci = (
            best_neighbor["pci"]
        )

        best_neighbor_rsrp = (
            best_neighbor["rsrp"]
        )

        best_neighbor_rsrq = (
            best_neighbor["rsrq"]
        )

        best_neighbor_sinr = (
            best_neighbor["sinr"]
        )


        features[
            "best_neighbor_index"
        ] = best_neighbor_index

        features[
            "best_neighbor_pci"
        ] = best_neighbor_pci

        features[
            "best_neighbor_rsrp"
        ] = best_neighbor_rsrp

        features[
            "best_neighbor_rsrp_delta"
        ] = (
            best_neighbor_rsrp
            - serving_rsrp
            if serving_rsrp is not None
            else None
        )


        if (
            best_neighbor_rsrq is not None
            and serving_rsrq is not None
        ):

            features[
                "best_neighbor_rsrq_delta"
            ] = (
                best_neighbor_rsrq
                - serving_rsrq
            )

        else:

            features[
                "best_neighbor_rsrq_delta"
            ] = None


        if (
            best_neighbor_sinr is not None
            and serving_sinr is not None
        ):

            features[
                "best_neighbor_sinr_delta"
            ] = (
                best_neighbor_sinr
                - serving_sinr
            )

        else:

            features[
                "best_neighbor_sinr_delta"
            ] = None


    else:

        features[
            "best_neighbor_index"
        ] = None

        features[
            "best_neighbor_pci"
        ] = None

        features[
            "best_neighbor_rsrp"
        ] = None

        features[
            "best_neighbor_rsrp_delta"
        ] = None

        features[
            "best_neighbor_rsrq_delta"
        ] = None

        features[
            "best_neighbor_sinr_delta"
        ] = None


    # ======================================================
    # NEIGHBOR AGGREGATE FEATURES
    # ======================================================

    rsrp_values = [
        n["rsrp"]
        for n in neighbors
        if n["rsrp"] is not None
    ]

    rsrq_values = [
        n["rsrq"]
        for n in neighbors
        if n["rsrq"] is not None
    ]

    sinr_values = [
        n["sinr"]
        for n in neighbors
        if n["sinr"] is not None
    ]


    # ------------------------------------------------------
    # RSRP
    # ------------------------------------------------------

    if rsrp_values:

        features[
            "neighbor_rsrp_max"
        ] = max(rsrp_values)

        features[
            "neighbor_rsrp_min"
        ] = min(rsrp_values)

        features[
            "neighbor_rsrp_mean"
        ] = (
            sum(rsrp_values)
            / len(rsrp_values)
        )

        features[
            "neighbor_rsrp_range"
        ] = (
            max(rsrp_values)
            - min(rsrp_values)
        )

    else:

        features[
            "neighbor_rsrp_max"
        ] = None

        features[
            "neighbor_rsrp_min"
        ] = None

        features[
            "neighbor_rsrp_mean"
        ] = None

        features[
            "neighbor_rsrp_range"
        ] = None


    # ------------------------------------------------------
    # RSRQ
    # ------------------------------------------------------

    if rsrq_values:

        features[
            "neighbor_rsrq_max"
        ] = max(rsrq_values)

        features[
            "neighbor_rsrq_min"
        ] = min(rsrq_values)

        features[
            "neighbor_rsrq_mean"
        ] = (
            sum(rsrq_values)
            / len(rsrq_values)
        )

        features[
            "neighbor_rsrq_range"
        ] = (
            max(rsrq_values)
            - min(rsrq_values)
        )

    else:

        features[
            "neighbor_rsrq_max"
        ] = None

        features[
            "neighbor_rsrq_min"
        ] = None

        features[
            "neighbor_rsrq_mean"
        ] = None

        features[
            "neighbor_rsrq_range"
        ] = None


    # ------------------------------------------------------
    # SINR
    # ------------------------------------------------------

    if sinr_values:

        features[
            "neighbor_sinr_max"
        ] = max(sinr_values)

        features[
            "neighbor_sinr_min"
        ] = min(sinr_values)

        features[
            "neighbor_sinr_mean"
        ] = (
            sum(sinr_values)
            / len(sinr_values)
        )

        features[
            "neighbor_sinr_range"
        ] = (
            max(sinr_values)
            - min(sinr_values)
        )

    else:

        features[
            "neighbor_sinr_max"
        ] = None

        features[
            "neighbor_sinr_min"
        ] = None

        features[
            "neighbor_sinr_mean"
        ] = None

        features[
            "neighbor_sinr_range"
        ] = None


    # ======================================================
    # SERVING VS BEST NEIGHBOR
    # ======================================================

    if (
        serving_rsrp is not None
        and features["neighbor_rsrp_max"] is not None
    ):

        features[
            "best_neighbor_rsrp_delta"
        ] = (
            features["neighbor_rsrp_max"]
            - serving_rsrp
        )


    if (
        serving_rsrq is not None
        and features["neighbor_rsrq_max"] is not None
    ):

        features[
            "best_neighbor_rsrq_delta"
        ] = (
            features["neighbor_rsrq_max"]
            - serving_rsrq
        )


    if (
        serving_sinr is not None
        and features["neighbor_sinr_max"] is not None
    ):

        features[
            "best_neighbor_sinr_delta"
        ] = (
            features["neighbor_sinr_max"]
            - serving_sinr
        )


    return features


# ==========================================================
# PROCESS ONE DATASET
# ==========================================================

def process_dataset(
    input_file,
    output_file,
    dataset_name
):

    print()
    print("=" * 80)
    print(
        f"PROCESSING {dataset_name}"
    )
    print("=" * 80)

    rows = []

    with open(
        input_file,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        input_fields = reader.fieldnames

        for row in reader:

            rows.append(row)


    print(
        f"Input rows : {len(rows):,}"
    )


    # ------------------------------------------------------
    # CREATE FEATURES
    # ------------------------------------------------------

    feature_rows = []

    for row in rows:

        engineered = engineer_row(row)

        # Keep the original identification,
        # timestamp, serving cell and target information.
        #
        # We deliberately keep the target columns in the
        # resulting dataset so that modeling scripts can
        # select the correct target later.

        output_row = {}

        # --------------------------------------------------
        # IDENTIFICATION / TIME
        # --------------------------------------------------

        output_row[
            "measurement_id"
        ] = row["measurement_id"]

        output_row[
            "source_file"
        ] = row["source_file"]

        output_row[
            "ue_id"
        ] = row["ue_id"]

        output_row[
            "gnb_ue_id"
        ] = row["gnb_ue_id"]

        output_row[
            "measurement_time"
        ] = row["measurement_time"]

        output_row[
            "serving_cell"
        ] = row["serving_cell"]


        # --------------------------------------------------
        # ENGINEERED FEATURES
        # --------------------------------------------------

        for key, value in engineered.items():

            output_row[key] = (
                format_number(value)
            )


        # --------------------------------------------------
        # TARGET / LABEL COLUMNS
        # --------------------------------------------------

        # These are retained for later modeling,
        # but MUST NOT be used as input features.

        target_columns = [
            "future_measurement_available",
            "future_measurement_time",
            "future_rsrp_raw",
            "future_rsrq_raw",
            "future_sinr_raw",
            "handover_within_3s",
            "future_handover_time",
            "future_target_cell",
            "future_target_pci"
        ]

        for column in target_columns:

            if column in row:

                output_row[column] = row[column]


        feature_rows.append(
            output_row
        )


    # ------------------------------------------------------
    # OUTPUT FIELD ORDER
    # ------------------------------------------------------

    base_columns = [
        "measurement_id",
        "source_file",
        "ue_id",
        "gnb_ue_id",
        "measurement_time",
        "serving_cell"
    ]


    engineered_columns = [
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

        "best_neighbor_index",
        "best_neighbor_pci",
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
        "neighbor_sinr_range"
    ]


    target_columns = [
        "future_measurement_available",
        "future_measurement_time",
        "future_rsrp_raw",
        "future_rsrq_raw",
        "future_sinr_raw",
        "handover_within_3s",
        "future_handover_time",
        "future_target_cell",
        "future_target_pci"
    ]


    output_fields = (
        base_columns
        + engineered_columns
        + target_columns
    )


    # ------------------------------------------------------
    # WRITE OUTPUT
    # ------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_fields
        )

        writer.writeheader()

        for row in feature_rows:

            writer.writerow(row)


    # ------------------------------------------------------
    # BASIC STATISTICS
    # ------------------------------------------------------

    total_rows = len(feature_rows)

    rows_with_neighbors = sum(
        int(
            row["neighbor_count"]
        ) > 0
        for row in feature_rows
    )

    rows_without_neighbors = (
        total_rows
        - rows_with_neighbors
    )


    print(
        f"Output rows          : "
        f"{total_rows:,}"
    )

    print(
        f"Output columns       : "
        f"{len(output_fields)}"
    )

    print(
        f"Rows with neighbors  : "
        f"{rows_with_neighbors:,}"
    )

    print(
        f"Rows without neighbors: "
        f"{rows_without_neighbors:,}"
    )

    print(
        f"Output file          : "
        f"{output_file}"
    )


# ==========================================================
# MAIN
# ==========================================================

print("=" * 80)
print("STEP 10 — FEATURE ENGINEERING")
print("=" * 80)

print()
print("Feature policy:")
print("- Raw radio measurements retained as encoded values.")
print("- No guessed physical-unit conversion.")
print("- No future information used as a feature.")
print("- No handover ground-truth information used as a feature.")
print("- Same feature definitions applied to all splits.")
print()


# ----------------------------------------------------------
# PROCESS TRAIN
# ----------------------------------------------------------

process_dataset(
    TRAIN_INPUT,
    TRAIN_OUTPUT,
    "TRAIN"
)


# ----------------------------------------------------------
# PROCESS VALIDATION
# ----------------------------------------------------------

process_dataset(
    VALIDATION_INPUT,
    VALIDATION_OUTPUT,
    "VALIDATION"
)


# ----------------------------------------------------------
# PROCESS TEST
# ----------------------------------------------------------

process_dataset(
    TEST_INPUT,
    TEST_OUTPUT,
    "TEST"
)


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 80)
print("STEP 10 — FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print()
print("Created:")
print(
    TRAIN_OUTPUT
)

print(
    VALIDATION_OUTPUT
)

print(
    TEST_OUTPUT
)

print()
print("IMPORTANT:")
print(
    "Target columns are retained for later modeling, "
    "but they must not be supplied as model input features."
)

print()
print("=" * 80)