import json
import glob
import os
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

# PATHS

BASE_DIR = r"c:\5G_Network_Quality"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "measurements"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "measurements_clean.csv"
)

# SETTINGS
MAX_NEIGHBORS = 3

# CREATE OUTPUT DIRECTORY

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV COLUMNS

columns = [
    "measurement_id",
    "source_file",

    "ue_id",
    "gnb_ue_id",

    "measurement_timestamp",
    "publish_timestamp",
    "tl_publish_timestamp",

    "enb",
    "du",
    "top_level_cell",

    "serving_cell",

    "serving_rsrp_raw",
    "serving_rsrp_present",

    "serving_rsrq_raw",
    "serving_rsrq_present",

    "serving_sinr_raw",
    "serving_sinr_present",

    "number_of_neighbors",

    "neighbor_1_cell",
    "neighbor_1_pci",
    "neighbor_1_rsrp_raw",
    "neighbor_1_rsrp_present",
    "neighbor_1_rsrq_raw",
    "neighbor_1_rsrq_present",
    "neighbor_1_sinr_raw",
    "neighbor_1_sinr_present",

    "neighbor_2_cell",
    "neighbor_2_pci",
    "neighbor_2_rsrp_raw",
    "neighbor_2_rsrp_present",
    "neighbor_2_rsrq_raw",
    "neighbor_2_rsrq_present",
    "neighbor_2_sinr_raw",
    "neighbor_2_sinr_present",

    "neighbor_3_cell",
    "neighbor_3_pci",
    "neighbor_3_rsrp_raw",
    "neighbor_3_rsrp_present",
    "neighbor_3_rsrq_raw",
    "neighbor_3_rsrq_present",
    "neighbor_3_sinr_raw",
    "neighbor_3_sinr_present",
]

# HELPERS

def is_present(value):
    return str(value).lower() == "true"


def get_measurement_value(obj, field):
    """
    Return the raw value only when the corresponding
    Present field says the measurement is present.
    """

    value = obj.get(field)
    present = obj.get(field + "Present")

    if not is_present(present):
        return None

    return value


def empty_neighbor(index):
    return {
        f"neighbor_{index}_cell": None,
        f"neighbor_{index}_pci": None,

        f"neighbor_{index}_rsrp_raw": None,
        f"neighbor_{index}_rsrp_present": False,

        f"neighbor_{index}_rsrq_raw": None,
        f"neighbor_{index}_rsrq_present": False,

        f"neighbor_{index}_sinr_raw": None,
        f"neighbor_{index}_sinr_present": False,
    }

# PROCESS FILES

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "neigh_measurements_*.txt"
        )
    )
)


if not files:
    print("ERROR: No measurement files found.")
    sys.exit(1)


total_records = 0
malformed_records = 0
records_with_neighbors = 0


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as output_file:

    writer = csv.DictWriter(
        output_file,
        fieldnames=columns
    )

    writer.writeheader()


    for filepath in files:

        filename = os.path.basename(filepath)

        print(f"Processing: {filename}")

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as input_file:

            for line_number, line in enumerate(
                input_file,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                except json.JSONDecodeError:
                    malformed_records += 1
                    continue


                total_records += 1


                info = data.get(
                    "RrcMeasurementReportResultInfo",
                    {}
                )

                serving = info.get(
                    "ServingCellInfo",
                    {}
                )

                # BASIC INFORMATION

                ue_id = data.get("UE")

                gnb_ue_id = info.get(
                    "GnbCuCpUeId"
                )

                # TIMESTAMPS

                measurement_timestamp = info.get(
                    "Timestamp"
                )

                publish_timestamp = data.get(
                    "timestamp"
                )

                tl_publish_timestamp = data.get(
                    "tlpublishTime__"
                )

                # SERVING CELL

                serving_nr_cgi = serving.get(
                    "NrCgi",
                    {}
                )

                serving_cell = serving_nr_cgi.get(
                    "NrCellId"
                )

                # SERVING RADIO MEASUREMENTS

                serving_rsrp = get_measurement_value(
                    serving,
                    "SsbRsrpResult"
                )

                serving_rsrq = get_measurement_value(
                    serving,
                    "SsbRsrqResult"
                )

                serving_sinr = get_measurement_value(
                    serving,
                    "SsbSinrResult"
                )

                # NEIGHBORS

                cell_info = info.get(
                    "CellInfo",
                    {}
                )

                neighbor_items = []

                if isinstance(cell_info, dict):

                    for key, wrapper in cell_info.items():

                        if not isinstance(wrapper, dict):
                            continue

                        neighbor = wrapper.get(
                            "NeighbourCellInfo",
                            {}
                        )

                        if not isinstance(neighbor, dict):
                            continue

                        pci = wrapper.get(
                            "PhyCellId"
                        )

                        neighbor_rsrp = get_measurement_value(
                            neighbor,
                            "SsbRsrpResult"
                        )

                        neighbor_rsrq = get_measurement_value(
                            neighbor,
                            "SsbRsrqResult"
                        )

                        neighbor_sinr = get_measurement_value(
                            neighbor,
                            "SsbSinrResult"
                        )


                        neighbor_items.append({
                            "pci": pci,

                            "nr_cell_id": neighbor.get(
                                "NrCgi",
                                {}
                            ).get(
                                "NrCellId"
                            ),

                            "rsrp": neighbor_rsrp,
                            "rsrp_present": is_present(
                                neighbor.get(
                                    "SsbRsrpResultPresent"
                                )
                            ),

                            "rsrq": neighbor_rsrq,
                            "rsrq_present": is_present(
                                neighbor.get(
                                    "SsbRsrqResultPresent"
                                )
                            ),

                            "sinr": neighbor_sinr,
                            "sinr_present": is_present(
                                neighbor.get(
                                    "SsbSinrResultPresent"
                                )
                            ),
                        })


                number_of_neighbors = len(
                    neighbor_items
                )


                if number_of_neighbors > 0:
                    records_with_neighbors += 1

                # CREATE ROW

                row = {
                    "measurement_id": total_records,
                    "source_file": filename,

                    "ue_id": ue_id,
                    "gnb_ue_id": gnb_ue_id,

                    "measurement_timestamp":
                        measurement_timestamp,

                    "publish_timestamp":
                        publish_timestamp,

                    "tl_publish_timestamp":
                        tl_publish_timestamp,

                    "enb": data.get("ENB"),
                    "du": data.get("DU"),
                    "top_level_cell": data.get("CELL"),

                    "serving_cell": serving_cell,

                    "serving_rsrp_raw":
                        serving_rsrp,

                    "serving_rsrp_present":
                        serving_rsrp is not None,

                    "serving_rsrq_raw":
                        serving_rsrq,

                    "serving_rsrq_present":
                        serving_rsrq is not None,

                    "serving_sinr_raw":
                        serving_sinr,

                    "serving_sinr_present":
                        serving_sinr is not None,

                    "number_of_neighbors":
                        number_of_neighbors,
                }

                # ADD EMPTY NEIGHBOR COLUMNS

                for i in range(
                    1,
                    MAX_NEIGHBORS + 1
                ):
                    row.update(
                        empty_neighbor(i)
                    )

                # ADD ACTUAL NEIGHBORS

                for i, neighbor in enumerate(
                    neighbor_items[:MAX_NEIGHBORS],
                    start=1
                ):

                    row[
                        f"neighbor_{i}_cell"
                    ] = neighbor["pci"]

                    row[
                        f"neighbor_{i}_pci"
                    ] = neighbor["pci"]

                    row[
                        f"neighbor_{i}_rsrp_raw"
                    ] = neighbor["rsrp"]

                    row[
                        f"neighbor_{i}_rsrp_present"
                    ] = neighbor["rsrp_present"]

                    row[
                        f"neighbor_{i}_rsrq_raw"
                    ] = neighbor["rsrq"]

                    row[
                        f"neighbor_{i}_rsrq_present"
                    ] = neighbor["rsrq_present"]

                    row[
                        f"neighbor_{i}_sinr_raw"
                    ] = neighbor["sinr"]

                    row[
                        f"neighbor_{i}_sinr_present"
                    ] = neighbor["sinr_present"]


                writer.writerow(row)

# SUMMARY

print()
print("MEASUREMENT CLEANING COMPLETE")

print(f"Total records processed : {total_records}")
print(f"Malformed records       : {malformed_records}")
print(f"Records with neighbors  : {records_with_neighbors}")

print()
print(f"Output file:")
print(OUTPUT_FILE)

print()
print("Columns:")
print(len(columns))