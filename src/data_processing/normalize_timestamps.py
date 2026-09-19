import csv
import os
from datetime import datetime, timezone
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

MEASUREMENT_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_clean.csv"
)

HANDOVER_INPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "handovers_clean.csv"
)

MEASUREMENT_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_normalized.csv"
)

HANDOVER_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "handovers_normalized.csv"
)


def microseconds_to_datetime(value):

    if value is None or value == "":
        return ""

    timestamp = int(value)

    dt = datetime.fromtimestamp(
        timestamp / 1_000_000,
        tz=timezone.utc
    )

    return dt.isoformat()


def nanoseconds_to_datetime(value):

    if value is None or value == "":
        return ""

    timestamp = int(value)

    dt = datetime.fromtimestamp(
        timestamp / 1_000_000_000,
        tz=timezone.utc
    )

    return dt.isoformat()


def normalize_measurements():

    print()
    print("NORMALIZING MEASUREMENT TIMESTAMPS")

    with open(
        MEASUREMENT_INPUT,
        "r",
        encoding="utf-8",
        newline=""
    ) as input_file:

        reader = csv.DictReader(input_file)

        fieldnames = list(reader.fieldnames)

        if "measurement_datetime" not in fieldnames:
            fieldnames.append("measurement_datetime")

        if "publish_datetime" not in fieldnames:
            fieldnames.append("publish_datetime")

        if "tl_publish_datetime" not in fieldnames:
            fieldnames.append("tl_publish_datetime")

        with open(
            MEASUREMENT_OUTPUT,
            "w",
            encoding="utf-8",
            newline=""
        ) as output_file:

            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            count = 0

            for row in reader:

                row["measurement_datetime"] = (
                    microseconds_to_datetime(
                        row["measurement_timestamp"]
                    )
                )

                row["publish_datetime"] = (
                    nanoseconds_to_datetime(
                        row["publish_timestamp"]
                    )
                )

                row["tl_publish_datetime"] = (
                    microseconds_to_datetime(
                        row["tl_publish_timestamp"]
                    )
                )

                writer.writerow(row)

                count += 1

    print(f"Records processed : {count}")
    print()
    print("Output:")
    print(MEASUREMENT_OUTPUT)


def normalize_handovers():

    print()
    print("NORMALIZING HANDOVER TIMESTAMPS")

    with open(
        HANDOVER_INPUT,
        "r",
        encoding="utf-8",
        newline=""
    ) as input_file:

        reader = csv.DictReader(input_file)

        fieldnames = list(reader.fieldnames)

        if "handover_datetime" not in fieldnames:
            fieldnames.append("handover_datetime")

        if "publish_datetime" not in fieldnames:
            fieldnames.append("publish_datetime")

        if "tl_publish_datetime" not in fieldnames:
            fieldnames.append("tl_publish_datetime")

        with open(
            HANDOVER_OUTPUT,
            "w",
            encoding="utf-8",
            newline=""
        ) as output_file:

            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            count = 0

            for row in reader:

                # Dedicated handover timestamp is blank in the supplied dataset, so leave this blank.

                row["handover_datetime"] = ""

                row["publish_datetime"] = (
                    nanoseconds_to_datetime(
                        row["publish_timestamp"]
                    )
                )

                row["tl_publish_datetime"] = (
                    microseconds_to_datetime(
                        row["tl_publish_timestamp"]
                    )
                )

                writer.writerow(row)

                count += 1

    print(f"Records processed : {count}")
    print()
    print("Output:")
    print(HANDOVER_OUTPUT)


normalize_measurements()
normalize_handovers()

print()
print("TIMESTAMP NORMALIZATION COMPLETE")