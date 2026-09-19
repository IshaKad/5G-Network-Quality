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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_datetime(value):
    if not value:
        return None

    return datetime.fromisoformat(value)


def get_neighbor_pcis(row):
    pcis = []

    for i in range(1, 4):

        value = row.get(
            f"neighbor_{i}_pci"
        )

        if value not in (None, ""):
            pcis.append(int(value))

    return pcis


# ------------------------------------------------------------
# Load measurements
# ------------------------------------------------------------

print("=" * 80)
print("LOADING MEASUREMENTS")
print("=" * 80)

measurements_by_ue = {}

measurement_count = 0

with open(
    MEASUREMENT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        ue = row["ue_id"]

        row["_dt"] = parse_datetime(
            row["measurement_datetime"]
        )

        measurements_by_ue.setdefault(
            ue,
            []
        ).append(row)

        measurement_count += 1


print(
    f"Measurements loaded : {measurement_count}"
)

print(
    f"Unique UEs          : "
    f"{len(measurements_by_ue)}"
)


# ------------------------------------------------------------
# Load handovers
# ------------------------------------------------------------

print()
print("=" * 80)
print("ANALYZING HANDOVERS")
print("=" * 80)

total_handovers = 0
matched_handovers = 0

source_matches = 0
target_pci_matches = 0

time_gaps = []

examples = []


with open(
    HANDOVER_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for handover in reader:

        total_handovers += 1

        ue = handover["ue_id"]

        handover_dt = parse_datetime(
            handover["tl_publish_datetime"]
        )

        source_cell = handover[
            "source_cell"
        ]

        target_pci = handover[
            "target_pci"
        ]

        if not handover_dt:
            continue

        if ue not in measurements_by_ue:
            continue

        ue_measurements = measurements_by_ue[ue]

        # ----------------------------------------------------
        # Find the latest measurement before the handover
        # ----------------------------------------------------

        previous_measurement = None

        for measurement in ue_measurements:

            if measurement["_dt"] < handover_dt:
                previous_measurement = measurement

            else:
                break

        if previous_measurement is None:
            continue

        matched_handovers += 1

        measurement_dt = previous_measurement[
            "_dt"
        ]

        gap_seconds = (
            handover_dt - measurement_dt
        ).total_seconds()

        time_gaps.append(
            gap_seconds
        )

        # ----------------------------------------------------
        # Source-cell consistency
        # ----------------------------------------------------

        measurement_serving_cell = (
            previous_measurement[
                "serving_cell"
            ]
        )

        if (
            measurement_serving_cell
            == source_cell
        ):
            source_matches += 1

        # ----------------------------------------------------
        # Target PCI consistency
        # ----------------------------------------------------

        neighbor_pcis = get_neighbor_pcis(
            previous_measurement
        )

        target_pci_int = None

        if target_pci not in (None, ""):
            target_pci_int = int(target_pci)

        target_found = (
            target_pci_int in neighbor_pcis
        )

        if target_found:
            target_pci_matches += 1

        # ----------------------------------------------------
        # Save examples
        # ----------------------------------------------------

        if len(examples) < 10:

            examples.append({
                "ue": ue,
                "measurement_time": measurement_dt,
                "handover_time": handover_dt,
                "gap": gap_seconds,
                "measurement_serving": (
                    measurement_serving_cell
                ),
                "source": source_cell,
                "neighbor_pcis": neighbor_pcis,
                "target_cell": handover[
                    "target_cell"
                ],
                "target_pci": target_pci,
                "target_found": target_found
            })


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 80)
print("ASSOCIATION SUMMARY")
print("=" * 80)

print(
    f"Total handovers              : "
    f"{total_handovers}"
)

print(
    f"Handovers matched to UE      : "
    f"{matched_handovers}"
)

print(
    f"Source-cell matches          : "
    f"{source_matches}"
)

print(
    f"Target PCI found in neighbors: "
    f"{target_pci_matches}"
)


if time_gaps:

    time_gaps_sorted = sorted(
        time_gaps
    )

    n = len(time_gaps_sorted)

    median = time_gaps_sorted[
        n // 2
    ]

    print()
    print("Measurement → handover gap:")

    print(
        f"Minimum : "
        f"{min(time_gaps_sorted):.6f} sec"
    )

    print(
        f"Median  : "
        f"{median:.6f} sec"
    )

    print(
        f"Maximum : "
        f"{max(time_gaps_sorted):.6f} sec"
    )


# ------------------------------------------------------------
# Examples
# ------------------------------------------------------------

print()
print("=" * 80)
print("EXAMPLE ASSOCIATIONS")
print("=" * 80)

for i, example in enumerate(
    examples,
    start=1
):

    print()
    print(f"Example {i}")

    print(
        f"UE                 : "
        f"{example['ue']}"
    )

    print(
        f"Measurement time   : "
        f"{example['measurement_time']}"
    )

    print(
        f"Handover time      : "
        f"{example['handover_time']}"
    )

    print(
        f"Time gap           : "
        f"{example['gap']:.6f} sec"
    )

    print(
        f"Measurement serving: "
        f"{example['measurement_serving']}"
    )

    print(
        f"Handover source    : "
        f"{example['source']}"
    )

    print(
        f"Neighbor PCIs      : "
        f"{example['neighbor_pcis']}"
    )

    print(
        f"Handover target    : "
        f"{example['target_cell']}"
    )

    print(
        f"Target PCI         : "
        f"{example['target_pci']}"
    )

    print(
        f"Target PCI found   : "
        f"{example['target_found']}"
    )


print()
print("=" * 80)
print("HANDOVER ASSOCIATION ANALYSIS COMPLETE")
print("=" * 80)