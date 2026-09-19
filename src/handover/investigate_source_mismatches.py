import csv
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

MEASUREMENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_normalized.csv"
)

HANDOVER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "handovers_normalized.csv"
)


def parse_datetime(value):
    if not value:
        return None
    return datetime.fromisoformat(value)


def get_neighbor_pcis(row):
    pcis = []

    for i in range(1, 4):
        value = row.get(f"neighbor_{i}_pci")

        if value not in (None, ""):
            pcis.append(int(value))

    return pcis

print("LOADING MEASUREMENTS")

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
        row["_dt"] = parse_datetime(row["measurement_datetime"])

        measurements_by_ue.setdefault(ue, []).append(row)


print(f"Unique UEs loaded : {len(measurements_by_ue)}")


print()
print("FINDING SOURCE-CELL MISMATCHES")

mismatch_count = 0
shown = 0

# How many measurements before/after the handover we inspect
WINDOW = 3


with open(
    HANDOVER_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for handover in reader:

        ue = handover["ue_id"]
        handover_dt = parse_datetime(
            handover["tl_publish_datetime"]
        )

        source_cell = handover["source_cell"]
        target_cell = handover["target_cell"]
        target_pci = handover["target_pci"]

        if not handover_dt:
            continue

        if ue not in measurements_by_ue:
            continue

        ue_measurements = measurements_by_ue[ue]

        # Find position of the last measurement before handover
        previous_index = None

        for i, measurement in enumerate(ue_measurements):

            if measurement["_dt"] < handover_dt:
                previous_index = i
            else:
                break

        if previous_index is None:
            continue

        previous_measurement = ue_measurements[previous_index]

        # Check whether the closest previous measurement matches the handover source.
        if previous_measurement["serving_cell"] == source_cell:
            continue

        mismatch_count += 1

        if shown >= 20:
            continue

        shown += 1

        print()
        print(f"Mismatch #{shown}")

        print(f"UE             : {ue}")
        print(f"Handover time  : {handover_dt}")
        print(f"Handover       : {source_cell} -> {target_cell}")
        print(f"Target PCI     : {target_pci}")

        print()
        print("Measurements around handover:")
        print()

        start = max(0, previous_index - WINDOW)
        end = min(
            len(ue_measurements),
            previous_index + WINDOW + 2
        )

        for j in range(start, end):

            measurement = ue_measurements[j]

            dt = measurement["_dt"]

            gap = (
                handover_dt - dt
            ).total_seconds()

            direction = (
                "BEFORE"
                if dt < handover_dt
                else "AFTER"
            )

            marker = ""

            if j == previous_index:
                marker = " <-- closest BEFORE"

            print(
                f"{direction:6s} | "
                f"{dt} | "
                f"gap={gap:+.6f}s | "
                f"serving={measurement['serving_cell']} | "
                f"neighbors={get_neighbor_pcis(measurement)}"
                f"{marker}"
            )


print()
print("SUMMARY")

print(f"Source-cell mismatches found : {mismatch_count}")
print(f"Detailed examples shown      : {min(shown, 20)}")

print()
print("SOURCE MISMATCH INVESTIGATION COMPLETE")
