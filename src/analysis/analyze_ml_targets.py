import csv
import os
import sys
from collections import Counter, defaultdict
from statistics import median

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_targets.csv"
)


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
        rows.append(row)


print(f"Total samples : {len(rows):,}")

# BASIC TARGET COUNTS

print()
print("TARGET DISTRIBUTION")

future_available = sum(
    row["future_measurement_available"] == "1"
    for row in rows
)

handover_positive = sum(
    row["handover_within_3s"] == "1"
    for row in rows
)

handover_negative = (
    len(rows)
    - handover_positive
)

print(
    f"Future measurement available : "
    f"{future_available:,}"
)

print(
    f"Future measurement unavailable : "
    f"{len(rows) - future_available:,}"
)

print()

print(
    f"Handover positive : "
    f"{handover_positive:,}"
)

print(
    f"Handover negative : "
    f"{handover_negative:,}"
)

print(
    f"Positive rate     : "
    f"{handover_positive / len(rows) * 100:.4f}%"
)

# HANDOVER TARGET CELL DISTRIBUTION

print()
print("FUTURE TARGET CELL DISTRIBUTION")

target_cells = Counter()

target_pcis = Counter()

for row in rows:

    if row["handover_within_3s"] == "1":

        target_cells[
            row["future_target_cell"]
        ] += 1

        target_pcis[
            row["future_target_pci"]
        ] += 1


print()
print("Target cells:")

for cell, count in sorted(
    target_cells.items(),
    key=lambda x: (-x[1], x[0])
):

    print(
        f"Cell {cell:<5} : {count:,}"
    )


print()
print("Target PCIs:")

for pci, count in sorted(
    target_pcis.items(),
    key=lambda x: (-x[1], x[0])
):

    print(
        f"PCI {pci:<5} : {count:,}"
    )


# HANDOVER TIME-TO-EVENT

print()
print("TIME TO HANDOVER")

# Calculate this from the ISO timestamps.

from datetime import datetime

time_to_handover = []

for row in rows:

    if row["handover_within_3s"] != "1":
        continue

    measurement_time = datetime.fromisoformat(
        row["measurement_time"]
    )

    handover_time = datetime.fromisoformat(
        row["future_handover_time"]
    )

    delta = (
        handover_time
        - measurement_time
    ).total_seconds()

    time_to_handover.append(delta)


if time_to_handover:

    print(
        f"Minimum : "
        f"{min(time_to_handover):.6f} sec"
    )

    print(
        f"Median  : "
        f"{median(time_to_handover):.6f} sec"
    )

    print(
        f"Maximum : "
        f"{max(time_to_handover):.6f} sec"
    )


# POSITIVE SAMPLES BY TIME-TO-HANDOVER RANGE

print()
print("POSITIVE SAMPLES BY TIME TO HANDOVER")

ranges = {
    "0-0.5 sec": 0,
    "0.5-1.0 sec": 0,
    "1.0-1.5 sec": 0,
    "1.5-2.0 sec": 0,
    "2.0-2.5 sec": 0,
    "2.5-3.0 sec": 0
}

for value in time_to_handover:

    if value <= 0.5:
        ranges["0-0.5 sec"] += 1

    elif value <= 1.0:
        ranges["0.5-1.0 sec"] += 1

    elif value <= 1.5:
        ranges["1.0-1.5 sec"] += 1

    elif value <= 2.0:
        ranges["1.5-2.0 sec"] += 1

    elif value <= 2.5:
        ranges["2.0-2.5 sec"] += 1

    else:
        ranges["2.5-3.0 sec"] += 1


for label, count in ranges.items():

    print(
        f"{label:<15} : {count:,}"
    )


# CURRENT SERVING CELL VS TARGET CELL

print()
print("CURRENT SERVING CELL → FUTURE TARGET CELL")

transitions = Counter()

for row in rows:

    if row["handover_within_3s"] != "1":
        continue

    source = row["serving_cell"]
    target = row["future_target_cell"]

    transitions[
        (source, target)
    ] += 1


for (source, target), count in sorted(
    transitions.items(),
    key=lambda x: (-x[1], x[0])
):

    print(
        f"{source} -> {target} : {count:,}"
    )


# FUTURE QUALITY AVAILABILITY

print()
print("FUTURE QUALITY AVAILABILITY")

for column in [
    "future_rsrp_raw",
    "future_rsrq_raw",
    "future_sinr_raw"
]:

    available = sum(
        row[column] != ""
        for row in rows
    )

    print(
        f"{column:<20} : "
        f"{available:,} / {len(rows):,}"
        f" ({available / len(rows) * 100:.2f}%)"
    )


print()
print("ML TARGET ANALYSIS COMPLETE")
