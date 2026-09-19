import json
import glob
import os
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

INPUT_DIR = os.path.join(BASE_DIR, "data", "raw", "handovers")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "handovers_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

columns = [
    "handover_id",
    "source_file",

    "ue_id",
    "gnb_ue_id",

    "handover_timestamp",
    "publish_timestamp",
    "tl_publish_timestamp",

    "source_cell",
    "target_cell",
    "target_pci",

    "target_du_id",
    "target_du_guid",

    "a3_offset",
    "hysteresis",
    "time_to_trigger",
    "trigger_quantity",
]

def get_nested(obj, *keys):
    current = obj

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "handover_events_*.txt")
    )
)

if not files:
    print("ERROR: No handover files found.")
    sys.exit(1)


total_records = 0
malformed_records = 0

source_missing = 0
target_missing = 0
target_pci_missing = 0

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

                event = data.get(
                    "HandoverEventIndicationData",
                    {}
                )

                # Basic identifiers

                ue_id = data.get("UE")

                gnb_ue_id = event.get(
                    "GnbCuUeF1apId"
                )
                # Timestamps

                handover_timestamp = event.get(
                    "Timestamp"
                )

                publish_timestamp = data.get(
                    "timestamp"
                )

                tl_publish_timestamp = data.get(
                    "tlpublishTime__"
                )

                # Source cell

                source_cell = get_nested(
                    event,
                    "SourceNrCgi",
                    "NrCellId"
                )

                # Target cell information

                target_cell_info = get_nested(
                    event,
                    "TargetCell",
                    "CellInfo"
                )

                if not isinstance(
                    target_cell_info,
                    dict
                ):
                    target_cell_info = {}

                target_cell = get_nested(
                    target_cell_info,
                    "NrCgi",
                    "NrCellId"
                )

                target_pci = target_cell_info.get(
                    "Pci"
                )

                target_du_id = target_cell_info.get(
                    "DuId"
                )

                target_du_guid = get_nested(
                    event,
                    "TargetCell",
                    "DuGuid"
                )

                # Measurement configuration

                rrc_config = event.get(
                    "RrcMeasConfigComplete",
                    {}
                )

                rptcfg_list = rrc_config.get(
                    "RptcfgToaddmodList",
                    {}
                )

                a3_offset = None
                hysteresis = None
                time_to_trigger = None
                trigger_quantity = None

                if isinstance(
                    rptcfg_list,
                    dict
                ):

                    # The report configuration may be stored
                    # under one or more numeric keys.
                    for _, config in rptcfg_list.items():

                        if not isinstance(
                            config,
                            dict
                        ):
                            continue

                        if a3_offset is None:
                            a3_offset = config.get(
                                "A3Offset"
                            )

                        if hysteresis is None:
                            hysteresis = config.get(
                                "Hysteresis"
                            )

                        if time_to_trigger is None:
                            time_to_trigger = config.get(
                                "TimeToTrigger"
                            )

                        if trigger_quantity is None:
                            trigger_quantity = config.get(
                                "TriggerQuantity"
                            )

                # Missing-field tracking

                if source_cell is None:
                    source_missing += 1

                if target_cell is None:
                    target_missing += 1

                if target_pci is None:
                    target_pci_missing += 1

                # Write row

                row = {
                    "handover_id": total_records,
                    "source_file": filename,

                    "ue_id": ue_id,
                    "gnb_ue_id": gnb_ue_id,

                    "handover_timestamp": handover_timestamp,
                    "publish_timestamp": publish_timestamp,
                    "tl_publish_timestamp": tl_publish_timestamp,

                    "source_cell": source_cell,
                    "target_cell": target_cell,
                    "target_pci": target_pci,

                    "target_du_id": target_du_id,
                    "target_du_guid": target_du_guid,

                    "a3_offset": a3_offset,
                    "hysteresis": hysteresis,
                    "time_to_trigger": time_to_trigger,
                    "trigger_quantity": trigger_quantity,
                }

                writer.writerow(row)


print()
print("HANDOVER CLEANING COMPLETE")

print(f"Total records processed : {total_records}")
print(f"Malformed records       : {malformed_records}")
print(f"Missing source cell     : {source_missing}")
print(f"Missing target cell     : {target_missing}")
print(f"Missing target PCI      : {target_pci_missing}")

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print("Columns:")
print(len(columns))