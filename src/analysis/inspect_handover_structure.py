import json
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

# PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HANDOVER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handovers"
)

# JSON READER
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

# RECURSIVE KEY INSPECTOR
def inspect_structure(obj, path="", depth=0, max_depth=4):

    if depth > max_depth:
        return

    if isinstance(obj, dict):

        for key, value in obj.items():

            current_path = (
                f"{path}.{key}"
                if path
                else key
            )

            print(
                "  " * depth
                + current_path
                + f"   [{type(value).__name__}]"
            )

            inspect_structure(
                value,
                current_path,
                depth + 1,
                max_depth
            )

    elif isinstance(obj, list):

        print(
            "  " * depth
            + f"{path} -> LIST ({len(obj)} items)"
        )

        if obj:

            inspect_structure(
                obj[0],
                path + "[0]",
                depth + 1,
                max_depth
            )

# MAIN

def inspect_handovers():

    files = sorted(
        HANDOVER_DIR.glob("*.txt")
    )

    print("             HANDOVER STRUCTURE INSPECTOR")

    total_records = 0

    top_level_keys = Counter()

    sample_records = []

    # READ ALL FILES
    for file_path in files:

        print(f"\nReading: {file_path.name}")

        records = read_json_records(file_path)

        print(
            f"  Records: {len(records)}"
        )

        total_records += len(records)

        for record in records:

            for key in record.keys():
                top_level_keys[key] += 1

            if len(sample_records) < 3:
                sample_records.append(record)

    # SUMMARY

    print("\n")
    print("SUMMARY")

    print(
        f"\nTotal handover records: "
        f"{total_records}"
    )

    print("\nTop-level keys:")

    for key, count in top_level_keys.items():

        print(
            f"  {key} -> {count} records"
        )

    # SAMPLE RECORDS

    print("\n")
    print("SAMPLE HANDOVER RECORDS")

    for index, record in enumerate(
        sample_records,
        start=1
    ):

        print(
            f"SAMPLE {index} "
        )

        print(
            json.dumps(
                record,
                indent=2
            )
        )

    # STRUCTURE OF FIRST RECORD

    if sample_records:

        print("\n")
        print("FIRST RECORD KEY STRUCTURE")

        inspect_structure(
            sample_records[0]
        )

    print("                     COMPLETE")

if __name__ == "__main__":
    inspect_handovers()