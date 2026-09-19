import json
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEASUREMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "measurements"
)

def read_json_records(file_path):

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    decoder = json.JSONDecoder()
    position = 0

    while position < len(content):

        while position < len(content) and content[position].isspace():
            position += 1

        if position >= len(content):
            break

        try:
            record, end_position = decoder.raw_decode(
                content,
                position
            )

            records.append(record)
            position = end_position

        except json.JSONDecodeError:

            next_object = content.find(
                "{",
                position + 1
            )

            if next_object == -1:
                break

            position = next_object

    return records

def inspect_neighbor_records():

    files = sorted(
        MEASUREMENT_DIR.glob("*.txt")
    )

    print("NEIGHBOR STRUCTURE INSPECTOR")

    total_neighbor_records = 0

    included_cells_counter = Counter()

    neighbor_key_counter = Counter()

    neighbor_samples = []

    for file_path in files:

        print(f"\nReading: {file_path.name}")

        records = read_json_records(file_path)

        for record in records:

            rrc_info = record.get(
                "RrcMeasurementReportResultInfo",
                {}
            )

            number_of_cells = rrc_info.get(
                "NumberOfIncludedCells",
                0
            )

            included_cells_counter[number_of_cells] += 1

            if number_of_cells <= 0:
                continue

            total_neighbor_records += 1

            for key in rrc_info.keys():
                neighbor_key_counter[key] += 1

            if len(neighbor_samples) < 5:
                neighbor_samples.append(record)

    print("\n")
    print("SUMMARY")

    print(
        f"\nRecords containing neighbors: "
        f"{total_neighbor_records}"
    )

    print("\nNumberOfIncludedCells distribution:")

    for count, frequency in sorted(
        included_cells_counter.items()
    ):
        print(
            f"  {count} neighbors/cells -> "
            f"{frequency} records"
        )

    print("\nKeys found inside RrcMeasurementReportResultInfo:")

    for key, count in neighbor_key_counter.items():
        print(
            f"  {key} -> {count}"
        )

    print("SAMPLE NEIGHBOR RECORDS")

    for index, record in enumerate(
        neighbor_samples,
        start=1
    ):

        print(
            f"\nSAMPLE {index} "
        )

        rrc_info = record.get(
            "RrcMeasurementReportResultInfo",
            {}
        )

        print(
            json.dumps(
                rrc_info,
                indent=2
            )
        )

    print("\n")
    print("COMPLETE")

if __name__ == "__main__":
    inspect_neighbor_records()