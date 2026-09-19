import csv
import os
from datetime import datetime, timezone
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

MEASUREMENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_clean.csv"
)

HANDOVER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "handovers_clean.csv"
)


def micros_to_datetime(value):
    return datetime.fromtimestamp(
        int(value) / 1_000_000,
        tz=timezone.utc
    )


def analyze_measurements():

    print("=" * 80)
    print("MEASUREMENT TIME ANALYSIS")
    print("=" * 80)

    count = 0
    min_ts = None
    max_ts = None

    previous_ts = None
    out_of_order = 0

    first_rows = []
    last_rows = []

    with open(
        MEASUREMENT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            value = row["measurement_timestamp"]

            if not value:
                continue

            ts = int(value)

            count += 1

            if min_ts is None or ts < min_ts:
                min_ts = ts

            if max_ts is None or ts > max_ts:
                max_ts = ts

            if previous_ts is not None and ts < previous_ts:
                out_of_order += 1

            previous_ts = ts

            if len(first_rows) < 3:
                first_rows.append(row)

            last_rows.append(row)

            if len(last_rows) > 3:
                last_rows.pop(0)

    print(f"Records with timestamp : {count}")
    print(f"Out-of-order records   : {out_of_order}")

    print()
    print(f"Minimum timestamp:")
    print(min_ts)
    print(micros_to_datetime(min_ts))

    print()
    print(f"Maximum timestamp:")
    print(max_ts)
    print(micros_to_datetime(max_ts))

    duration_seconds = (
        max_ts - min_ts
    ) / 1_000_000

    print()
    print(
        f"Total time span        : "
        f"{duration_seconds:.2f} seconds"
    )

    print(
        f"Total time span        : "
        f"{duration_seconds / 60:.2f} minutes"
    )

    print(
        f"Total time span        : "
        f"{duration_seconds / 3600:.2f} hours"
    )

    print()
    print("First 3 records:")

    for row in first_rows:
        print(
            row["measurement_id"],
            row["ue_id"],
            row["serving_cell"],
            row["measurement_timestamp"]
        )

    print()
    print("Last 3 records:")

    for row in last_rows:
        print(
            row["measurement_id"],
            row["ue_id"],
            row["serving_cell"],
            row["measurement_timestamp"]
        )


def analyze_handovers():

    print()
    print("HANDOVER TIME ANALYSIS")

    count = 0
    min_ts = None
    max_ts = None

    previous_ts = None
    out_of_order = 0

    with open(
        HANDOVER_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            value = row["tl_publish_timestamp"]

            if not value:
                continue

            ts = int(value)

            count += 1

            if min_ts is None or ts < min_ts:
                min_ts = ts

            if max_ts is None or ts > max_ts:
                max_ts = ts

            if previous_ts is not None and ts < previous_ts:
                out_of_order += 1

            previous_ts = ts

    print(f"Records with timestamp : {count}")
    print(f"Out-of-order records   : {out_of_order}")

    print()
    print("Minimum timestamp:")
    print(min_ts)
    print(micros_to_datetime(min_ts))

    print()
    print("Maximum timestamp:")
    print(max_ts)
    print(micros_to_datetime(max_ts))

    duration_seconds = (
        max_ts - min_ts
    ) / 1_000_000

    print()
    print(
        f"Total time span        : "
        f"{duration_seconds:.2f} seconds"
    )

    print(
        f"Total time span        : "
        f"{duration_seconds / 60:.2f} minutes"
    )

    print(
        f"Total time span        : "
        f"{duration_seconds / 3600:.2f} hours"
    )


analyze_measurements()
analyze_handovers()

print()
print("TIME RANGE ANALYSIS COMPLETE")