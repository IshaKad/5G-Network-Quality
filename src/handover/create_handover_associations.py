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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "handover_associations.csv"
)


def parse_datetime(value):
    if not value:
        return None

    return datetime.fromisoformat(value)


def get_neighbor_values(row, index):
    """Return neighbor PCI and radio measurements for one neighbor."""

    return {
        f"neighbor_{index}_pci":
            row.get(f"neighbor_{index}_pci", ""),

        f"neighbor_{index}_rsrp_raw":
            row.get(f"neighbor_{index}_rsrp_raw", ""),

        f"neighbor_{index}_rsrq_raw":
            row.get(f"neighbor_{index}_rsrq_raw", ""),

        f"neighbor_{index}_sinr_raw":
            row.get(f"neighbor_{index}_sinr_raw", ""),

        f"neighbor_{index}_rsrp_present":
            row.get(f"neighbor_{index}_rsrp_present", ""),

        f"neighbor_{index}_rsrq_present":
            row.get(f"neighbor_{index}_rsrq_present", ""),

        f"neighbor_{index}_sinr_present":
            row.get(f"neighbor_{index}_sinr_present", "")
    }

print("LOADING MEASUREMENTS")

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


print(f"Measurements loaded : {measurement_count}")
print(f"Unique UEs          : {len(measurements_by_ue)}")


print()
print("CREATING HANDOVER ASSOCIATIONS")

associations = []

total_handovers = 0
matched_handovers = 0

source_matches = 0
target_pci_matches = 0

time_gaps = []


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

        if not handover_dt:
            continue

        if ue not in measurements_by_ue:
            continue

        ue_measurements = measurements_by_ue[ue]

        # Find the latest measurement BEFORE the recorded handover temporal reference.

        previous_measurement = None

        for measurement in ue_measurements:

            if measurement["_dt"] < handover_dt:
                previous_measurement = measurement
            else:
                break

        if previous_measurement is None:
            continue

        matched_handovers += 1

        measurement_dt = previous_measurement["_dt"]

        gap_seconds = (
            handover_dt - measurement_dt
        ).total_seconds()

        time_gaps.append(gap_seconds)

        # Source-cell consistency

        measurement_serving_cell = (
            previous_measurement["serving_cell"]
        )

        source_cell = handover["source_cell"]

        source_cell_matches = (
            measurement_serving_cell == source_cell
        )

        if source_cell_matches:
            source_matches += 1

        # Target PCI consistency

        neighbor_pcis = []

        for i in range(1, 4):

            pci = previous_measurement.get(
                f"neighbor_{i}_pci"
            )

            if pci not in (None, ""):
                neighbor_pcis.append(int(pci))

        target_pci = handover["target_pci"]

        target_pci_found = False

        if target_pci not in (None, ""):

            target_pci_int = int(target_pci)

            if target_pci_int in neighbor_pcis:
                target_pci_found = True
                target_pci_matches += 1

        # Build final association row

        association = {

            # Handover event identity

            "handover_id":
                handover["handover_id"],

            "source_file":
                handover["source_file"],

            "ue_id":
                handover["ue_id"],

            "gnb_ue_id":
                handover["gnb_ue_id"],

            # Handover event timing

            "handover_time":
                handover["tl_publish_datetime"],

            "handover_publish_time":
                handover["publish_datetime"],

            # Actual handover ground truth

            "source_cell":
                handover["source_cell"],

            "target_cell":
                handover["target_cell"],

            "target_pci":
                handover["target_pci"],

            "target_du_id":
                handover["target_du_id"],

            "target_du_guid":
                handover["target_du_guid"],

            # Pre-event measurement context

            "measurement_id":
                previous_measurement["measurement_id"],

            "measurement_time":
                previous_measurement["measurement_datetime"],

            "measurement_to_handover_gap_seconds":
                f"{gap_seconds:.6f}",

            "measurement_serving_cell":
                previous_measurement["serving_cell"],

            # Serving-cell radio measurements

            "serving_rsrp_raw":
                previous_measurement["serving_rsrp_raw"],

            "serving_rsrq_raw":
                previous_measurement["serving_rsrq_raw"],

            "serving_sinr_raw":
                previous_measurement["serving_sinr_raw"],

            "serving_rsrp_present":
                previous_measurement["serving_rsrp_present"],

            "serving_rsrq_present":
                previous_measurement["serving_rsrq_present"],

            "serving_sinr_present":
                previous_measurement["serving_sinr_present"],

            # Number of neighbors

            "number_of_neighbors":
                previous_measurement["number_of_neighbors"],

            # Neighbor measurements

            **get_neighbor_values(previous_measurement, 1),
            **get_neighbor_values(previous_measurement, 2),
            **get_neighbor_values(previous_measurement, 3),

            # Association diagnostics

            "source_cell_matches":
                source_cell_matches,

            "target_pci_found_in_neighbors":
                target_pci_found
        }

        associations.append(association)

fieldnames = [
    "handover_id",
    "source_file",
    "ue_id",
    "gnb_ue_id",

    "handover_time",
    "handover_publish_time",

    "source_cell",
    "target_cell",
    "target_pci",
    "target_du_id",
    "target_du_guid",

    "measurement_id",
    "measurement_time",
    "measurement_to_handover_gap_seconds",
    "measurement_serving_cell",

    "serving_rsrp_raw",
    "serving_rsrq_raw",
    "serving_sinr_raw",

    "serving_rsrp_present",
    "serving_rsrq_present",
    "serving_sinr_present",

    "number_of_neighbors",

    "neighbor_1_pci",
    "neighbor_1_rsrp_raw",
    "neighbor_1_rsrq_raw",
    "neighbor_1_sinr_raw",
    "neighbor_1_rsrp_present",
    "neighbor_1_rsrq_present",
    "neighbor_1_sinr_present",

    "neighbor_2_pci",
    "neighbor_2_rsrp_raw",
    "neighbor_2_rsrq_raw",
    "neighbor_2_sinr_raw",
    "neighbor_2_rsrp_present",
    "neighbor_2_rsrq_present",
    "neighbor_2_sinr_present",

    "neighbor_3_pci",
    "neighbor_3_rsrp_raw",
    "neighbor_3_rsrq_raw",
    "neighbor_3_sinr_raw",
    "neighbor_3_rsrp_present",
    "neighbor_3_rsrq_present",
    "neighbor_3_sinr_present",

    "source_cell_matches",
    "target_pci_found_in_neighbors"
]


print()
print("WRITING OUTPUT")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in associations:
        writer.writerow(row)


# SUMMARY

print(f"Output file : {OUTPUT_FILE}")
print(f"Rows written: {len(associations)}")

print()
print("Association statistics:")
print(f"Total handovers              : {total_handovers}")
print(f"Matched handovers            : {matched_handovers}")
print(f"Source-cell matches          : {source_matches}")
print(f"Target PCI found in neighbors: {target_pci_matches}")

if matched_handovers > 0:

    print()
    print("Percentages:")

    print(
        f"UE/measurement association   : "
        f"{matched_handovers / total_handovers * 100:.2f}%"
    )

    print(
        f"Source-cell agreement        : "
        f"{source_matches / matched_handovers * 100:.2f}%"
    )

    print(
        f"Target PCI agreement         : "
        f"{target_pci_matches / matched_handovers * 100:.2f}%"
    )


if time_gaps:

    sorted_gaps = sorted(time_gaps)

    n = len(sorted_gaps)

    median_gap = sorted_gaps[n // 2]

    print()
    print("Measurement → handover gap:")

    print(
        f"Minimum : {min(sorted_gaps):.6f} sec"
    )

    print(
        f"Median  : {median_gap:.6f} sec"
    )

    print(
        f"Maximum : {max(sorted_gaps):.6f} sec"
    )


print()
print("HANDOVER ASSOCIATION DATASET CREATED")