import json
import glob
import os
from collections import Counter

import sys
sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = r"c:\5G_Network_Quality"
MEASUREMENT_DIR = os.path.join(BASE_DIR, "data", "raw", "measurements")


files = sorted(
    glob.glob(os.path.join(MEASUREMENT_DIR, "neigh_measurements_*.txt"))
)


def inspect_file(filepath):
    print("\n" + "=" * 90)
    print(f"FILE: {os.path.basename(filepath)}")
    print("=" * 90)

    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            info = data.get("RrcMeasurementReportResultInfo", {})

            serving = info.get("ServingCellInfo", {})

            print("\nTop-level keys:")
            print(list(data.keys()))

            print("\nMeasurement info keys:")
            print(list(info.keys()))

            print("\nServingCellInfo keys:")
            print(list(serving.keys()))

            print("\nServing RSRP:")
            print(
                "Value:",
                serving.get("SsbRsrpResult"),
                "Present:",
                serving.get("SsbRsrpResultPresent")
            )

            print("\nServing RSRQ:")
            print(
                "Value:",
                serving.get("SsbRsrqResult"),
                "Present:",
                serving.get("SsbRsrqResultPresent")
            )

            print("\nServing SINR:")
            print(
                "Value:",
                serving.get("SsbSinrResult"),
                "Present:",
                serving.get("SsbSinrResultPresent")
            )

            print("\nNumberOfIncludedCells:")
            print(info.get("NumberOfIncludedCells"))

            # Show first record with neighbors
            cell_info = info.get("CellInfo")

            if cell_info:
                print("\nNeighbor structure:")
                for index, neighbor_wrapper in cell_info.items():

                    neighbor = neighbor_wrapper.get(
                        "NeighbourCellInfo", {}
                    )

                    print(f"\nNeighbor {index}:")
                    print("Keys:", list(neighbor.keys()))

                    print(
                        "RSRP:",
                        neighbor.get("SsbRsrpResult"),
                        "| Present:",
                        neighbor.get("SsbRsrpResultPresent")
                    )

                    print(
                        "RSRQ:",
                        neighbor.get("SsbRsrqResult"),
                        "| Present:",
                        neighbor.get("SsbRsrqResultPresent")
                    )

                    print(
                        "SINR:",
                        neighbor.get("SsbSinrResult"),
                        "| Present:",
                        neighbor.get("SsbSinrResultPresent")
                    )

                    print(
                        "NrCgi:",
                        neighbor.get("NrCgi")
                    )

                    print(
                        "PhyCellId:",
                        neighbor_wrapper.get("PhyCellId")
                    )

            print("\nTimestamp:")
            print(info.get("Timestamp"))

            print("\nRaw record:")
            print(json.dumps(data, indent=2))

            # Only inspect the first record of this file
            break


for filepath in files:
    inspect_file(filepath)