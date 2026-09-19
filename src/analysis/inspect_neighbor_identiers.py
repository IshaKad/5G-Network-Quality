import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

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

def inspect_neighbor_identifiers():

    files = sorted(
        MEASUREMENT_DIR.glob("*.txt")
    )

    neighbor_nr_cell_ids = Counter()

    neighbor_phy_cell_ids = Counter()

    serving_cell_ids = Counter()

    serving_to_neighbor = Counter()

    neighbor_count_distribution = Counter()

    examples_by_neighbor_count = {}

    total_records = 0
    records_with_neighbors = 0

    for file_path in files:

        print(f"Reading: {file_path.name}")

        records = read_json_records(file_path)

        for record in records:

            total_records += 1

            rrc_info = record.get(
                "RrcMeasurementReportResultInfo",
                {}
            )

            serving_info = rrc_info.get(
                "ServingCellInfo",
                {}
            )

            serving_nr_cgi = serving_info.get(
                "NrCgi",
                {}
            )

            serving_cell = serving_nr_cgi.get(
                "NrCellId"
            )

            if serving_cell is not None:
                serving_cell_ids[serving_cell] += 1

            number_of_cells = rrc_info.get(
                "NumberOfIncludedCells",
                0
            )

            neighbor_count_distribution[
                number_of_cells
            ] += 1

            if number_of_cells <= 0:
                continue

            records_with_neighbors += 1

            cell_info = rrc_info.get(
                "CellInfo",
                {}
            )
            # Inspect every neighbor

            for index, neighbor_wrapper in cell_info.items():

                neighbor_info = neighbor_wrapper.get(
                    "NeighbourCellInfo",
                    {}
                )

                nr_cgi = neighbor_info.get(
                    "NrCgi",
                    {}
                )

                neighbor_nr_cell = nr_cgi.get(
                    "NrCellId"
                )

                phy_cell_id = neighbor_wrapper.get(
                    "PhyCellId"
                )

                if neighbor_nr_cell is not None:
                    neighbor_nr_cell_ids[
                        neighbor_nr_cell
                    ] += 1

                if phy_cell_id is not None:
                    neighbor_phy_cell_ids[
                        phy_cell_id
                    ] += 1

                serving_to_neighbor[
                    (
                        serving_cell,
                        neighbor_nr_cell,
                        phy_cell_id
                    )
                ] += 1

            if number_of_cells not in examples_by_neighbor_count:

                examples_by_neighbor_count[
                    number_of_cells
                ] = record

    print("\n")
    print("IDENTIFIER ANALYSIS")

    print(
        f"\nTotal measurement records: "
        f"{total_records}"
    )

    print(
        f"Records containing neighbors: "
        f"{records_with_neighbors}"
    )

    print("\n")
    print("SERVING NrCellId VALUES")

    for cell_id, count in sorted(
        serving_cell_ids.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"  NrCellId {cell_id} -> {count} records"
        )

    print("\n")
    print("NEIGHBOR NrCellId VALUES")

    for cell_id, count in sorted(
        neighbor_nr_cell_ids.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"  NrCellId {cell_id} -> {count} occurrences"
        )

    print("\n")
    print("NEIGHBOR PhyCellId VALUES")

    for phy_id, count in sorted(
        neighbor_phy_cell_ids.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"  PhyCellId {phy_id} -> {count} occurrences"
        )

    print("\n")
    print("SERVING -> NEIGHBOR IDENTIFIER RELATIONSHIPS")

    for (
        serving,
        neighbor_nr,
        phy
    ), count in sorted(
        serving_to_neighbor.items(),
        key=lambda x: (-x[1], str(x[0]))
    ):

        print(
            f"  Serving={serving} "
            f"-> NeighborNrCellId={neighbor_nr}, "
            f"PhyCellId={phy} "
            f"-> {count} occurrences"
        )

    print("\n")
    print("SAMPLE RECORDS")

    for neighbor_count in [1, 2, 3]:

        if neighbor_count not in examples_by_neighbor_count:
            continue

        record = examples_by_neighbor_count[
            neighbor_count
        ]

        print(
            f"{neighbor_count} NEIGHBOR(S) "
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
    inspect_neighbor_identifiers()