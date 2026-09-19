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
    "ml_targets.csv"
)

HORIZON_SECONDS = 3.0
TOLERANCE_SECONDS = 0.5


def parse_datetime(value):
    return datetime.fromisoformat(value)

print("LOADING MEASUREMENTS")

measurements_by_ue = {}
all_measurements = []

with open(
    MEASUREMENT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["_datetime"] = parse_datetime(
            row["measurement_datetime"]
        )

        ue = row["ue_id"]

        measurements_by_ue.setdefault(
            ue,
            []
        ).append(row)

        all_measurements.append(row)


for ue in measurements_by_ue:

    measurements_by_ue[ue].sort(
        key=lambda x: x["_datetime"]
    )


print(
    f"Measurements loaded : {len(all_measurements):,}"
)

print(
    f"Unique UEs          : {len(measurements_by_ue)}"
)


print()
print("LOADING HANDOVERS")

handovers_by_ue = {}

with open(
    HANDOVER_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["_datetime"] = parse_datetime(
            row["tl_publish_datetime"]
        )

        ue = row["ue_id"]

        handovers_by_ue.setdefault(
            ue,
            []
        ).append(row)


for ue in handovers_by_ue:

    handovers_by_ue[ue].sort(
        key=lambda x: x["_datetime"]
    )


print(
    f"Handovers loaded    : "
    f"{sum(len(v) for v in handovers_by_ue.values()):,}"
)

print(
    f"UEs with handovers  : "
    f"{len(handovers_by_ue)}"
)


print()
print("CREATING ML TARGETS")

output_rows = []

future_measurement_count = 0
handover_positive_count = 0
no_future_measurement_count = 0


for ue, measurements in measurements_by_ue.items():

    handovers = handovers_by_ue.get(
        ue,
        []
    )

    measurement_count = len(measurements)

    # Pointer for future handover search.
    handover_index = 0

    for i, current in enumerate(measurements):

        current_time = current["_datetime"]

        # FIND FUTURE MEASUREMENT AT ~3 SECONDS

        desired_time = (
            current_time.timestamp()
            + HORIZON_SECONDS
        )

        future_measurement = None

        # Start from the next measurement.

        for j in range(
            i + 1,
            min(i + 10, measurement_count)
        ):

            candidate = measurements[j]

            delta = (
                candidate["_datetime"]
                - current_time
            ).total_seconds()

            if (
                HORIZON_SECONDS - TOLERANCE_SECONDS
                <= delta
                <= HORIZON_SECONDS + TOLERANCE_SECONDS
            ):

                future_measurement = candidate
                break

            if delta > (
                HORIZON_SECONDS
                + TOLERANCE_SECONDS
            ):

                break


        if future_measurement is not None:

            future_measurement_count += 1

        else:

            no_future_measurement_count += 1

        # FIND NEXT HANDOVER WITHIN 3 SECONDS

        handover_within_3s = 0

        future_target_cell = ""
        future_target_pci = ""

        future_handover_time = ""

        # Advance pointer past handovers that are already before or at the current measurement time.

        while (
            handover_index < len(handovers)
            and handovers[handover_index]["_datetime"]
            <= current_time
        ):

            handover_index += 1


        if handover_index < len(handovers):

            next_handover = handovers[
                handover_index
            ]

            handover_delta = (
                next_handover["_datetime"]
                - current_time
            ).total_seconds()


            if (
                handover_delta > 0
                and handover_delta <= HORIZON_SECONDS
            ):

                handover_within_3s = 1

                future_target_cell = (
                    next_handover["target_cell"]
                )

                future_target_pci = (
                    next_handover["target_pci"]
                )

                future_handover_time = (
                    next_handover[
                        "tl_publish_datetime"
                    ]
                )

                handover_positive_count += 1

        # BUILD OUTPUT ROW

        output_row = {
            "measurement_id":
                current["measurement_id"],

            "source_file":
                current["source_file"],

            "ue_id":
                current["ue_id"],

            "gnb_ue_id":
                current["gnb_ue_id"],

            "measurement_time":
                current["measurement_datetime"],

            "serving_cell":
                current["serving_cell"],

            "serving_rsrp_raw":
                current["serving_rsrp_raw"],

            "serving_rsrq_raw":
                current["serving_rsrq_raw"],

            "serving_sinr_raw":
                current["serving_sinr_raw"],

            "serving_rsrp_present":
                current["serving_rsrp_present"],

            "serving_rsrq_present":
                current["serving_rsrq_present"],

            "serving_sinr_present":
                current["serving_sinr_present"],

            "number_of_neighbors":
                current["number_of_neighbors"],

            "neighbor_1_pci":
                current["neighbor_1_pci"],

            "neighbor_1_rsrp_raw":
                current["neighbor_1_rsrp_raw"],

            "neighbor_1_rsrq_raw":
                current["neighbor_1_rsrq_raw"],

            "neighbor_1_sinr_raw":
                current["neighbor_1_sinr_raw"],

            "neighbor_1_rsrp_present":
                current["neighbor_1_rsrp_present"],

            "neighbor_1_rsrq_present":
                current["neighbor_1_rsrq_present"],

            "neighbor_1_sinr_present":
                current["neighbor_1_sinr_present"],

            "neighbor_2_pci":
                current["neighbor_2_pci"],

            "neighbor_2_rsrp_raw":
                current["neighbor_2_rsrp_raw"],

            "neighbor_2_rsrq_raw":
                current["neighbor_2_rsrq_raw"],

            "neighbor_2_sinr_raw":
                current["neighbor_2_sinr_raw"],

            "neighbor_2_rsrp_present":
                current["neighbor_2_rsrp_present"],

            "neighbor_2_rsrq_present":
                current["neighbor_2_rsrq_present"],

            "neighbor_2_sinr_present":
                current["neighbor_2_sinr_present"],

            "neighbor_3_pci":
                current["neighbor_3_pci"],

            "neighbor_3_rsrp_raw":
                current["neighbor_3_rsrp_raw"],

            "neighbor_3_rsrq_raw":
                current["neighbor_3_rsrq_raw"],

            "neighbor_3_sinr_raw":
                current["neighbor_3_sinr_raw"],

            "neighbor_3_rsrp_present":
                current["neighbor_3_rsrp_present"],

            "neighbor_3_rsrq_present":
                current["neighbor_3_rsrq_present"],

            "neighbor_3_sinr_present":
                current["neighbor_3_sinr_present"],

            # FUTURE QUALITY TARGETS

            "future_measurement_available":
                int(future_measurement is not None),

            "future_measurement_time":
                (
                    future_measurement[
                        "measurement_datetime"
                    ]
                    if future_measurement is not None
                    else ""
                ),

            "future_rsrp_raw":
                (
                    future_measurement[
                        "serving_rsrp_raw"
                    ]
                    if future_measurement is not None
                    else ""
                ),

            "future_rsrq_raw":
                (
                    future_measurement[
                        "serving_rsrq_raw"
                    ]
                    if future_measurement is not None
                    else ""
                ),

            "future_sinr_raw":
                (
                    future_measurement[
                        "serving_sinr_raw"
                    ]
                    if future_measurement is not None
                    else ""
                ),

            # HANDOVER TARGETS

            "handover_within_3s":
                handover_within_3s,

            "future_handover_time":
                future_handover_time,

            "future_target_cell":
                future_target_cell,

            "future_target_pci":
                future_target_pci
        }

        output_rows.append(output_row)

# WRITE OUTPUT

fieldnames = list(output_rows[0].keys())

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

    writer.writerows(output_rows)

# SUMMARY

print()
print("TARGET CREATION SUMMARY")

print(
    f"Total ML samples             : "
    f"{len(output_rows):,}"
)

print(
    f"Future measurement available : "
    f"{future_measurement_count:,}"
)

print(
    f"No future measurement         : "
    f"{no_future_measurement_count:,}"
)

print(
    f"Handover-positive samples    : "
    f"{handover_positive_count:,}"
)

print(
    f"Handover-negative samples    : "
    f"{len(output_rows) - handover_positive_count:,}"
)


if output_rows:

    positive_rate = (
        handover_positive_count
        / len(output_rows)
        * 100
    )

    print(
        f"Handover positive rate      : "
        f"{positive_rate:.4f}%"
    )


print()
print(
    f"Output file:"
)

print(OUTPUT_FILE)

print()
print("STEP 8 TARGET CREATION COMPLETE")