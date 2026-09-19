import csv
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_targets.csv"
)

# A gap larger than this means we consider the
# measurements to belong to separate recording segments.
SEGMENT_GAP_SECONDS = 60


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


print(
    f"Total samples : {len(rows):,}"
)

# FIND RECORDING SEGMENTS

segments = []

current_segment = []

previous_time = None

for row in rows:

    current_time = row["_datetime"]

    if previous_time is None:

        current_segment = [row]

    else:

        gap = (
            current_time
            - previous_time
        ).total_seconds()

        if gap > SEGMENT_GAP_SECONDS:

            segments.append(
                current_segment
            )

            current_segment = [row]

        else:

            current_segment.append(row)

    previous_time = current_time


if current_segment:
    segments.append(current_segment)

# PRINT SEGMENTS

print()
print("RECORDING SEGMENTS")

print(
    f"Gap threshold : "
    f"{SEGMENT_GAP_SECONDS} seconds"
)

print(
    f"Segments found : "
    f"{len(segments)}"
)

print()

for index, segment in enumerate(
    segments,
    start=1
):

    start = segment[0]["_datetime"]
    end = segment[-1]["_datetime"]

    duration = (
        end - start
    ).total_seconds()

    samples = len(segment)

    positives = sum(
        row["handover_within_3s"] == "1"
        for row in segment
    )

    source_files = sorted(
        set(
            row["source_file"]
            for row in segment
        )
    )


    print(
        f"Segment {index}"
    )

    print(
        f"Start       : {start}"
    )

    print(
        f"End         : {end}"
    )

    print(
        f"Duration    : "
        f"{duration / 3600:.4f} hours"
    )

    print(
        f"Samples     : {samples:,}"
    )

    print(
        f"HO positives: {positives:,}"
    )

    if samples:

        print(
            f"HO rate     : "
            f"{positives / samples * 100:.4f}%"
        )

    print(
        f"Source files: "
        f"{', '.join(source_files)}"
    )


# GAPS BETWEEN SEGMENTS

print()
print("GAPS BETWEEN RECORDING SEGMENTS")

for i in range(
    len(segments) - 1
):

    end_current = (
        segments[i][-1]["_datetime"]
    )

    start_next = (
        segments[i + 1][0]["_datetime"]
    )

    gap = (
        start_next
        - end_current
    ).total_seconds()

    print(
        f"Segment {i + 1} → "
        f"Segment {i + 2} : "
        f"{gap:.3f} sec"
    )


print()
print("RECORDING SEGMENT ANALYSIS COMPLETE")