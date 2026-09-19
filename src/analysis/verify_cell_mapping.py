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

HANDOVER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handovers"
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

# MEASUREMENT CELL ANALYSIS

def inspect_measurements():

    print("\n")
    print("MEASUREMENT CELL MAPPING")

    cell_field_to_nr_cell = Counter()

    serving_to_phy = Counter()

    for file_path in sorted(
        MEASUREMENT_DIR.glob("*.txt")
    ):

        print(f"Reading measurement: {file_path.name}")

        records = read_json_records(file_path)

        for record in records:

            top_cell = record.get("CELL")

            rrc_info = record.get(
                "RrcMeasurementReportResultInfo",
                {}
            )

            serving_info = rrc_info.get(
                "ServingCellInfo",
                {}
            )

            serving_nr = (
                serving_info
                .get("NrCgi", {})
                .get("NrCellId")
            )

            if top_cell is not None and serving_nr is not None:

                cell_field_to_nr_cell[
                    (top_cell, serving_nr)
                ] += 1

            cell_info = rrc_info.get(
                "CellInfo",
                {}
            )

            for wrapper in cell_info.values():

                neighbor_info = wrapper.get(
                    "NeighbourCellInfo",
                    {}
                )

                phy_id = wrapper.get(
                    "PhyCellId"
                )

                if serving_nr is not None and phy_id is not None:

                    serving_to_phy[
                        (serving_nr, phy_id)
                    ] += 1

    print("\n")
    print("TOP-LEVEL CELL -> SERVING NrCellId")

    for (
        top_cell,
        nr_cell
    ), count in sorted(
        cell_field_to_nr_cell.items()
    ):

        print(
            f"  CELL={top_cell} "
            f"-> NrCellId={nr_cell} "
            f"-> {count} records"
        )

    print("\n")
    print("SERVING NrCellId -> NEIGHBOR PhyCellId")

    for (
        serving,
        phy
    ), count in sorted(
        serving_to_phy.items()
    ):

        print(
            f"  Serving={serving} "
            f"-> Neighbor PCI={phy} "
            f"-> {count} occurrences"
        )

def inspect_handovers():

    print("\n")
    print("HANDOVER CELL MAPPING")

    source_target = Counter()

    source_target_pci = Counter()

    source_cell_values = Counter()
    target_cell_values = Counter()

    for file_path in sorted(
        HANDOVER_DIR.glob("*.txt")
    ):

        print(f"Reading handover: {file_path.name}")

        records = read_json_records(file_path)

        for record in records:

            source = record.get(
                "SourceNrCgi",
                {}
            )

            source_nr = source.get(
                "NrCellId"
            )

            target = record.get(
                "TargetCell",
                {}
            )

            target_info = target.get(
                "CellInfo",
                {}
            )

            target_nr_cgi = target_info.get(
                "NrCgi",
                {}
            )

            target_nr = target_nr_cgi.get(
                "NrCellId"
            )

            target_pci = target_info.get(
                "PhyCellId"
            )

            if source_nr is not None:
                source_cell_values[source_nr] += 1

            if target_nr is not None:
                target_cell_values[target_nr] += 1

            if source_nr is not None and target_nr is not None:

                source_target[
                    (source_nr, target_nr)
                ] += 1

            if (
                source_nr is not None
                and target_pci is not None
            ):

                source_target_pci[
                    (source_nr, target_pci)
                ] += 1

    print("\n")
    print("HANDOVER SOURCE NrCellId")

    for cell, count in sorted(
        source_cell_values.items()
    ):

        print(
            f"  Source NrCellId={cell} "
            f"-> {count} handovers"
        )

    print("\n")
    print("HANDOVER TARGET NrCellId")

    for cell, count in sorted(
        target_cell_values.items()
    ):

        print(
            f"  Target NrCellId={cell} "
            f"-> {count} handovers"
        )

    print("\n")
    print("HANDOVER SOURCE -> TARGET NrCellId")

    for (
        source,
        target
    ), count in sorted(
        source_target.items()
    ):

        print(
            f"  {source} -> {target} "
            f"-> {count}"
        )

    print("\n")
    print("HANDOVER SOURCE -> TARGET PCI")

    for (
        source,
        pci
    ), count in sorted(
        source_target_pci.items()
    ):

        print(
            f"  Source={source} "
            f"-> Target PCI={pci} "
            f"-> {count}"
        )

if __name__ == "__main__":

    inspect_measurements()

    inspect_handovers()

    print("\n")
    print("COMPLETE")