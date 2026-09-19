import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

MEASUREMENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_normalized.csv"
)

HORIZONS = [1, 2, 3, 4, 5]

# Acceptable timing error around the desired horizon.
# Example:
# desired = t + 3 sec
# acceptable = approximately 2.5 to 3.5 sec
TOLERANCE = 0.5


def parse_datetime(value):
    from datetime import datetime
    return datetime.fromisoformat(value)


print("=" * 80)
print("LOADING MEASUREMENTS")
print("=" * 80)

measurements_by_ue = {}

with open(
    MEASUREMENT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        ue = row["ue_id"]

        dt = parse_datetime(
            row["measurement_datetime"]
        )

        measurements_by_ue.setdefault(
            ue,
            []
        ).append(dt)


print(f"Unique UEs loaded : {len(measurements_by_ue)}")


print()
print("=" * 80)
print("ANALYZING FUTURE HORIZON COVERAGE")
print("=" * 80)

results = {
    horizon: {
        "total": 0,
        "valid": 0
    }
    for horizon in HORIZONS
}


for ue, timestamps in measurements_by_ue.items():

    timestamps.sort()

    n = len(timestamps)

    for i, current_time in enumerate(timestamps):

        for horizon in HORIZONS:

            results[horizon]["total"] += 1

            desired_time = (
                current_time.timestamp()
                + horizon
            )

            # Search forward from current position.
            #
            # Since timestamps are approximately 1 second apart,
            # only a small number of records need to be checked.

            found = False

            for j in range(i + 1, min(i + 10, n)):

                future_time = timestamps[j]

                delta = (
                    future_time.timestamp()
                    - current_time.timestamp()
                )

                if (
                    horizon - TOLERANCE
                    <= delta
                    <= horizon + TOLERANCE
                ):
                    found = True
                    break

                if delta > horizon + TOLERANCE:
                    break

            if found:
                results[horizon]["valid"] += 1


print()

print(
    f"{'Horizon':<12}"
    f"{'Total':>12}"
    f"{'Valid':>12}"
    f"{'Coverage':>14}"
)

print("-" * 50)

for horizon in HORIZONS:

    total = results[horizon]["total"]
    valid = results[horizon]["valid"]

    coverage = (
        valid / total * 100
        if total
        else 0
    )

    print(
        f"{horizon} sec"
        f"{total:>12,}"
        f"{valid:>12,}"
        f"{coverage:>13.2f}%"
    )


print()
print("=" * 80)
print("FUTURE HORIZON COVERAGE ANALYSIS COMPLETE")
print("=" * 80)