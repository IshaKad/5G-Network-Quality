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

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "validation"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "test"
)

TRAIN_FILE = os.path.join(
    TRAIN_DIR,
    "ml_train.csv"
)

VALIDATION_FILE = os.path.join(
    VALIDATION_DIR,
    "ml_validation.csv"
)

TEST_FILE = os.path.join(
    TEST_DIR,
    "ml_test.csv"
)

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "split_summary.txt"
)

# SPLIT CONFIGURATION

TRAIN_SEGMENTS = {1, 2}
VALIDATION_SEGMENTS = {3}
TEST_SEGMENTS = {4, 5, 6, 7}

# Future prediction horizon.
PURGE_SECONDS = 3

# HELPERS

def parse_datetime(value):
    return datetime.fromisoformat(value)


def calculate_segments(rows):
    """
    Identify recording segments using gaps > 60 seconds.
    """

    SEGMENT_GAP_SECONDS = 60

    segments = []

    current_segment = []
    previous_time = None

    for row in rows:

        current_time = row["_datetime"]

        if previous_time is None:

            current_segment = [row]

        else:

            gap = (
                current_time - previous_time
            ).total_seconds()

            if gap > SEGMENT_GAP_SECONDS:

                segments.append(current_segment)

                current_segment = [row]

            else:

                current_segment.append(row)

        previous_time = current_time

    if current_segment:
        segments.append(current_segment)

    return segments


def write_csv(filename, rows, fieldnames):

    with open(
        filename,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            output_row = {
                key: row[key]
                for key in fieldnames
            }

            writer.writerow(output_row)


def calculate_statistics(rows):

    total = len(rows)

    positives = sum(
        row["handover_within_3s"] == "1"
        for row in rows
    )

    negatives = total - positives

    future_available = sum(
        row["future_measurement_available"] == "1"
        for row in rows
    )

    if total > 0:
        positive_rate = (
            positives / total * 100
        )
    else:
        positive_rate = 0

    return {
        "samples": total,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positive_rate,
        "future_available": future_available
    }

# LOAD DATA

print("LOADING ML TARGETS")

rows = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames

    for row in reader:

        row["_datetime"] = parse_datetime(
            row["measurement_time"]
        )

        rows.append(row)


rows.sort(
    key=lambda row: row["_datetime"]
)

print(
    f"Total samples : {len(rows):,}"
)

# IDENTIFY RECORDING SEGMENTS

segments = calculate_segments(rows)

print(
    f"Recording segments : {len(segments)}"
)

if len(segments) != 7:

    raise RuntimeError(
        "Expected exactly 7 recording segments. "
        f"Found {len(segments)}."
    )

# DETERMINE SPLIT BOUNDARIES

# Segment 2 is the last training segment.
# Segment 3 is validation.
# Segment 4 is the first test segment.

train_end = segments[1][-1]["_datetime"]

validation_end = segments[2][-1]["_datetime"]

validation_start = segments[2][0]["_datetime"]

test_start = segments[3][0]["_datetime"]


# Purge windows.

train_purge_start = (
    train_end - timedelta(
        seconds=PURGE_SECONDS
    )
)

validation_purge_start = (
    validation_end - timedelta(
        seconds=PURGE_SECONDS
    )
)

# CREATE SPLITS

train_rows = []
validation_rows = []
test_rows = []

purged_train = []
purged_validation = []


for row in rows:

    timestamp = row["_datetime"]

    # TRAIN

    if timestamp <= train_end:

        if timestamp >= train_purge_start:

            purged_train.append(row)

        else:

            train_rows.append(row)

    # VALIDATION

    elif (
        validation_start
        <= timestamp
        <= validation_end
    ):

        if timestamp >= validation_purge_start:

            purged_validation.append(row)

        else:

            validation_rows.append(row)

    # TEST

    elif timestamp >= test_start:

        test_rows.append(row)


# REMOVE INTERNAL HELPER COLUMN

output_fieldnames = [
    field
    for field in fieldnames
    if field != "_datetime"
]

# CREATE DIRECTORIES

os.makedirs(
    TRAIN_DIR,
    exist_ok=True
)

os.makedirs(
    VALIDATION_DIR,
    exist_ok=True
)

os.makedirs(
    TEST_DIR,
    exist_ok=True
)

# WRITE SPLIT FILES

write_csv(
    TRAIN_FILE,
    train_rows,
    output_fieldnames
)

write_csv(
    VALIDATION_FILE,
    validation_rows,
    output_fieldnames
)

write_csv(
    TEST_FILE,
    test_rows,
    output_fieldnames
)

# STATISTICS

train_stats = calculate_statistics(
    train_rows
)

validation_stats = calculate_statistics(
    validation_rows
)

test_stats = calculate_statistics(
    test_rows
)

# PRINT RESULTS

print()
print("TEMPORAL SPLIT RESULTS")

print()
print("TRAIN")

print(
    f"Segments             : 1, 2"
)

print(
    f"Samples              : "
    f"{train_stats['samples']:,}"
)

print(
    f"HO positives         : "
    f"{train_stats['positives']:,}"
)

print(
    f"HO negatives         : "
    f"{train_stats['negatives']:,}"
)

print(
    f"HO positive rate     : "
    f"{train_stats['positive_rate']:.4f}%"
)

print(
    f"Future quality avail.: "
    f"{train_stats['future_available']:,}"
)

print(
    f"Time start           : "
    f"{train_rows[0]['_datetime']}"
)

print(
    f"Time end             : "
    f"{train_rows[-1]['_datetime']}"
)


print()
print("VALIDATION")

print(
    f"Segment              : 3"
)

print(
    f"Samples              : "
    f"{validation_stats['samples']:,}"
)

print(
    f"HO positives         : "
    f"{validation_stats['positives']:,}"
)

print(
    f"HO negatives         : "
    f"{validation_stats['negatives']:,}"
)

print(
    f"HO positive rate     : "
    f"{validation_stats['positive_rate']:.4f}%"
)

print(
    f"Future quality avail.: "
    f"{validation_stats['future_available']:,}"
)

print(
    f"Time start           : "
    f"{validation_rows[0]['_datetime']}"
)

print(
    f"Time end             : "
    f"{validation_rows[-1]['_datetime']}"
)


print()
print("TEST")

print(
    f"Segments             : 4, 5, 6, 7"
)

print(
    f"Samples              : "
    f"{test_stats['samples']:,}"
)

print(
    f"HO positives         : "
    f"{test_stats['positives']:,}"
)

print(
    f"HO negatives         : "
    f"{test_stats['negatives']:,}"
)

print(
    f"HO positive rate     : "
    f"{test_stats['positive_rate']:.4f}%"
)

print(
    f"Future quality avail.: "
    f"{test_stats['future_available']:,}"
)

print(
    f"Time start           : "
    f"{test_rows[0]['_datetime']}"
)

print(
    f"Time end             : "
    f"{test_rows[-1]['_datetime']}"
)

# PURGE INFORMATION

print()
print("TEMPORAL PURGE")

print(
    f"Purge horizon : "
    f"{PURGE_SECONDS} seconds"
)

print(
    f"Training samples purged     : "
    f"{len(purged_train):,}"
)

print(
    f"Validation samples purged   : "
    f"{len(purged_validation):,}"
)

# CHECK FOR OVERLAP

print()
print("SPLIT BOUNDARY CHECK")

print(
    f"Train last sample      : "
    f"{train_rows[-1]['_datetime']}"
)

print(
    f"Validation first sample: "
    f"{validation_rows[0]['_datetime']}"
)

print(
    f"Validation last sample : "
    f"{validation_rows[-1]['_datetime']}"
)

print(
    f"Test first sample      : "
    f"{test_rows[0]['_datetime']}"
)


if (
    train_rows[-1]["_datetime"]
    >= validation_rows[0]["_datetime"]
):

    raise RuntimeError(
        "TRAIN / VALIDATION overlap detected."
    )


if (
    validation_rows[-1]["_datetime"]
    >= test_rows[0]["_datetime"]
):

    raise RuntimeError(
        "VALIDATION / TEST overlap detected."
    )


print()
print("No temporal overlap detected.")

# WRITE SUMMARY

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "5G NETWORK QUALITY PREDICTION\n"
        "TEMPORAL DATASET SPLIT SUMMARY\n"
        "\n"
    )

    f.write(
        "Split strategy:\n"
        "Train      = Recording Segments 1 + 2\n"
        "Validation = Recording Segment 3\n"
        "Test       = Recording Segments 4 + 5 + 6 + 7\n"
        "\n"
    )

    f.write(
        f"Purging horizon = {PURGE_SECONDS} seconds\n"
        "\n"
    )

    for name, stats in [
        ("TRAIN", train_stats),
        ("VALIDATION", validation_stats),
        ("TEST", test_stats)
    ]:

        f.write(
            f"{name}\n"
        )

        f.write(
            f"Samples = {stats['samples']}\n"
        )

        f.write(
            f"HO positives = {stats['positives']}\n"
        )

        f.write(
            f"HO negatives = {stats['negatives']}\n"
        )

        f.write(
            f"HO positive rate = "
            f"{stats['positive_rate']:.4f}%\n"
        )

        f.write(
            f"Future quality available = "
            f"{stats['future_available']}\n"
        )

        f.write("\n")

    f.write(
        f"Training samples purged = "
        f"{len(purged_train)}\n"
    )

    f.write(
        f"Validation samples purged = "
        f"{len(purged_validation)}\n"
    )

print()
print("FILES CREATED")

print(
    TRAIN_FILE
)

print(
    VALIDATION_FILE
)

print(
    TEST_FILE
)

print(
    SUMMARY_FILE
)

print()
print("STEP 9 SPLIT COMPLETE")