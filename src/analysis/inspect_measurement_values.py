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

def inspect_values():

    rsrp_counter = Counter()
    rsrq_counter = Counter()
    sinr_counter = Counter()

    neighbor_rsrp_counter = Counter()
    neighbor_rsrq_counter = Counter()
    neighbor_sinr_counter = Counter()

    total_records = 0
    neighbor_records = 0

    # READ

    for file_path in sorted(
        MEASUREMENT_DIR.glob("*.txt")
    ):

        print(f"Reading: {file_path.name}")

        records = read_json_records(file_path)

        for record in records:

            total_records += 1

            rrc = record.get(
                "RrcMeasurementReportResultInfo",
                {}
            )

            # SERVING CELL

            serving = rrc.get(
                "ServingCellInfo",
                {}
            )

            if serving.get("SsbRsrpResultPresent") == "true":
                rsrp_counter[
                    serving.get("SsbRsrpResult")
                ] += 1

            if serving.get("SsbRsrqResultPresent") == "true":
                rsrq_counter[
                    serving.get("SsbRsrqResult")
                ] += 1

            if serving.get("SsbSinrResultPresent") == "true":
                sinr_counter[
                    serving.get("SsbSinrResult")
                ] += 1

            # NEIGHBORS

            number_of_cells = rrc.get(
                "NumberOfIncludedCells",
                0
            )

            if number_of_cells <= 0:
                continue

            neighbor_records += 1

            cell_info = rrc.get(
                "CellInfo",
                {}
            )

            for wrapper in cell_info.values():

                neighbor = wrapper.get(
                    "NeighbourCellInfo",
                    {}
                )

                if neighbor.get(
                    "SsbRsrpResultPresent"
                ) == "true":

                    neighbor_rsrp_counter[
                        neighbor.get("SsbRsrpResult")
                    ] += 1

                if neighbor.get(
                    "SsbRsrqResultPresent"
                ) == "true":

                    neighbor_rsrq_counter[
                        neighbor.get("SsbRsrqResult")
                    ] += 1

                if neighbor.get(
                    "SsbSinrResultPresent"
                ) == "true":

                    neighbor_sinr_counter[
                        neighbor.get("SsbSinrResult")
                    ] += 1

    # FUNCTION FOR PRINTING DISTRIBUTION

    def print_distribution(title, counter):

        print("\n")
        print(title)

        values = sorted(
            value
            for value in counter
            if value is not None
        )

        if not values:
            print("No values found.")
            return

        print(
            f"Unique values: {len(values)}"
        )

        print(
            f"Minimum: {min(values)}"
        )

        print(
            f"Maximum: {max(values)}"
        )

        print("\nValue frequencies:")

        for value in values:

            print(
                f"  {value:>3} -> "
                f"{counter[value]}"
            )

    # SUMMARY

    print("\n")
    print("RADIO MEASUREMENT ANALYSIS")

    print(
        f"\nTotal records: {total_records}"
    )

    print(
        f"Records with neighbors: {neighbor_records}"
    )

    # SERVING

    print_distribution(
        "SERVING RSRP RAW VALUES",
        rsrp_counter
    )

    print_distribution(
        "SERVING RSRQ RAW VALUES",
        rsrq_counter
    )

    print_distribution(
        "SERVING SINR RAW VALUES",
        sinr_counter
    )

    # NEIGHBORS

    print_distribution(
        "NEIGHBOR RSRP RAW VALUES",
        neighbor_rsrp_counter
    )

    print_distribution(
        "NEIGHBOR RSRQ RAW VALUES",
        neighbor_rsrq_counter
    )

    print_distribution(
        "NEIGHBOR SINR RAW VALUES",
        neighbor_sinr_counter
    )

    print("\n")
    print("COMPLETE")


if __name__ == "__main__":
    inspect_values()