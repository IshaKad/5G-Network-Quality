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


def convert_timestamp(value):
    """
    Try interpreting a Unix timestamp according to
    its number of digits.

    16 digits -> microseconds
    19 digits -> nanoseconds
    """

    if value is None or value == "":
        return None

    value = int(value)

    digits = len(str(abs(value)))

    if digits >= 18:
        seconds = value / 1_000_000_000
        unit = "nanoseconds"
    elif digits >= 15:
        seconds = value / 1_000_000
        unit = "microseconds"
    elif digits >= 12:
        seconds = value / 1_000
        unit = "milliseconds"
    else:
        seconds = value
        unit = "seconds"

    dt = datetime.fromtimestamp(
        seconds,
        tz=timezone.utc
    )

    return dt, unit


def inspect_file(filepath, timestamp_columns, sample_count=10):

    print()
    print(f"FILE: {os.path.basename(filepath)}")

    rows = []

    with open(
        filepath,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            if i < sample_count:
                rows.append(row)
            else:
                break

    for row_number, row in enumerate(rows, start=1):

        print()
        print(f"Record {row_number}")

        for column in timestamp_columns:

            value = row.get(column)

            if value:

                try:
                    dt, unit = convert_timestamp(value)

                    print(
                        f"{column}: {value}"
                    )
                    print(
                        f"  interpreted as: {unit}"
                    )
                    print(
                        f"  UTC datetime   : {dt}"
                    )

                except Exception as e:

                    print(
                        f"{column}: {value}"
                    )
                    print(
                        f"  ERROR: {e}"
                    )

            else:

                print(
                    f"{column}: <blank>"
                )


inspect_file(
    MEASUREMENT_FILE,
    [
        "measurement_timestamp",
        "publish_timestamp",
        "tl_publish_timestamp"
    ]
)

inspect_file(
    HANDOVER_FILE,
    [
        "handover_timestamp",
        "publish_timestamp",
        "tl_publish_timestamp"
    ]
)

print()
print("TIMESTAMP VALIDATION COMPLETE")