import csv
import os
import sys
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_targets.csv"
)


def parse_datetime(value):
    return datetime.fromisoformat(value)

print("LOADING ML TARGETS")

rows = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["_datetime"] = parse_datetime(
            row["measurement_time"]
        )

        rows.append(row)


rows.sort(
    key=lambda row: row["_datetime"]
)


print(f"Total samples : {len(rows):,}")

# DATASET TIME RANGE

start_time = rows[0]["_datetime"]
end_time = rows[-1]["_datetime"]

total_duration = (
    end_time - start_time
)

print()
print("DATASET TIME RANGE")

print(
    f"Start : {start_time}"
)

print(
    f"End   : {end_time}"
)

print(
    f"Duration : "
    f"{total_duration.total_seconds() / 3600:.4f} hours"
)

# TIME-BASED BOUNDARIES

train_boundary = (
    start_time
    + total_duration * 0.70
)

validation_boundary = (
    start_time
    + total_duration * 0.85
)

print()
print("CHRONOLOGICAL SPLIT BOUNDARIES")

print(
    f"Train ends before       : "
    f"{train_boundary}"
)

print(
    f"Validation ends before  : "
    f"{validation_boundary}"
)

print(
    f"Test starts at          : "
    f"{validation_boundary}"
)

# COUNT SAMPLES

train_rows = []
validation_rows = []
test_rows = []

for row in rows:

    dt = row["_datetime"]

    if dt < train_boundary:

        train_rows.append(row)

    elif dt < validation_boundary:

        validation_rows.append(row)

    else:

        test_rows.append(row)

# SUMMARY FUNCTION

def summarize(name, dataset):

    total = len(dataset)

    positives = sum(
        row["handover_within_3s"] == "1"
        for row in dataset
    )

    negatives = total - positives

    future_available = sum(
        row["future_measurement_available"] == "1"
        for row in dataset
    )

    print()
    print(name)

    print(
        f"Samples                  : {total:,}"
    )

    print(
        f"Handover positives       : {positives:,}"
    )

    print(
        f"Handover negatives       : {negatives:,}"
    )

    print(
        f"Handover positive rate   : "
        f"{positives / total * 100:.4f}%"
        if total
        else "Handover positive rate   : N/A"
    )

    print(
        f"Future quality available : "
        f"{future_available:,}"
    )

    if dataset:

        print(
            f"Start                    : "
            f"{dataset[0]['_datetime']}"
        )

        print(
            f"End                      : "
            f"{dataset[-1]['_datetime']}"
        )


print()
print("SPLIT SUMMARY")

summarize(
    "TRAIN",
    train_rows
)

summarize(
    "VALIDATION",
    validation_rows
)

summarize(
    "TEST",
    test_rows
)


print()
print("SPLIT BOUNDARY ANALYSIS COMPLETE")