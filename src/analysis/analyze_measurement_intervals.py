import csv
import os
import sys
from datetime import datetime
from statistics import median

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"c:\5G_Network_Quality"

MEASUREMENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "measurements_normalized.csv"
)


def parse_datetime(value):
    return datetime.fromisoformat(value)


print("=" * 80)
print("LOADING MEASUREMENTS")
print("=" * 80)

measurements_by_ue = {}

with open(
    MEASUREMENT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        ue = row["ue_id"]

        dt = parse_datetime(
            row["measurement_datetime"]
        )

        measurements_by_ue.setdefault(
            ue,
            []
        ).append(dt)


print(f"Unique UEs loaded : {len(measurements_by_ue)}")


print()
print("=" * 80)
print("ANALYZING MEASUREMENT INTERVALS")
print("=" * 80)


all_intervals = []
ue_statistics = []


for ue, timestamps in measurements_by_ue.items():

    timestamps.sort()

    intervals = []

    for i in range(1, len(timestamps)):

        gap = (
            timestamps[i] - timestamps[i - 1]
        ).total_seconds()

        if gap >= 0:
            intervals.append(gap)
            all_intervals.append(gap)

    if intervals:

        ue_statistics.append({
            "ue": ue,
            "count": len(timestamps),
            "median_interval": median(intervals),
            "minimum_interval": min(intervals),
            "maximum_interval": max(intervals)
        })


print()
print("Overall measurement intervals:")
print(f"Total intervals : {len(all_intervals):,}")

if all_intervals:

    sorted_intervals = sorted(all_intervals)

    n = len(sorted_intervals)

    def percentile(values, p):
        index = int(p * (len(values) - 1))
        return values[index]

    print(
        f"Minimum         : {min(sorted_intervals):.6f} sec"
    )

    print(
        f"25th percentile : "
        f"{percentile(sorted_intervals, 0.25):.6f} sec"
    )

    print(
        f"Median          : "
        f"{percentile(sorted_intervals, 0.50):.6f} sec"
    )

    print(
        f"75th percentile : "
        f"{percentile(sorted_intervals, 0.75):.6f} sec"
    )

    print(
        f"90th percentile : "
        f"{percentile(sorted_intervals, 0.90):.6f} sec"
    )

    print(
        f"95th percentile : "
        f"{percentile(sorted_intervals, 0.95):.6f} sec"
    )

    print(
        f"99th percentile : "
        f"{percentile(sorted_intervals, 0.99):.6f} sec"
    )

    print(
        f"Maximum         : "
        f"{max(sorted_intervals):.6f} sec"
    )


print()
print("=" * 80)
print("EXAMPLE UE-LEVEL INTERVALS")
print("=" * 80)

for stats in ue_statistics[:15]:

    print(
        f"UE {stats['ue']:>5} | "
        f"records={stats['count']:>5} | "
        f"median={stats['median_interval']:.6f}s | "
        f"min={stats['minimum_interval']:.6f}s | "
        f"max={stats['maximum_interval']:.6f}s"
    )


print()
print("=" * 80)
print("MEASUREMENT INTERVAL ANALYSIS COMPLETE")
print("=" * 80)