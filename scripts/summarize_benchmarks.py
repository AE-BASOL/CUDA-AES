#!/usr/bin/env python3
"""Summarize Phase 3 CUDA-AES raw benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {
    "schema_version",
    "benchmark_run_id",
    "timing_scope",
    "device",
    "cipher",
    "block_size",
    "run_index",
    "run_count",
    "time_ms",
    "GiB/s",
    "operation",
}

GROUP_FIELDS = ("device", "cipher", "operation", "block_size", "timing_scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary tables from CUDA-AES Phase 3 raw benchmark CSV output."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("bench/thr_gpu.csv"), Path("bench/thr_cpu.csv")],
        help="Raw benchmark CSV files. Defaults to bench/thr_gpu.csv and bench/thr_cpu.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("bench/summary.md"),
        help="Summary Markdown output path. Defaults to bench/summary.md.",
    )
    return parser.parse_args()


def validate_fields(path: Path, fieldnames: Iterable[str] | None) -> None:
    if not fieldnames:
        raise SystemExit(f"{path}: missing CSV header")
    missing = sorted(REQUIRED_FIELDS - set(fieldnames))
    if missing:
        raise SystemExit(f"{path}: missing required columns: {', '.join(missing)}")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path}: file not found")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            validate_fields(path, reader.fieldnames)
            for line_no, row in enumerate(reader, start=2):
                try:
                    float(row["time_ms"])
                    float(row["GiB/s"])
                    int(row["block_size"])
                    int(row["run_index"])
                    int(row["run_count"])
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid numeric field: {exc}") from exc
                rows.append(row)
    if not rows:
        raise SystemExit("no benchmark rows found")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups[key].append(row)

    summary_rows: list[dict[str, str]] = []
    for key in sorted(groups.keys(), key=lambda item: (item[0], item[1], item[2], int(item[3]), item[4])):
        grouped = groups[key]
        times = [float(row["time_ms"]) for row in grouped]
        throughputs = [float(row["GiB/s"]) for row in grouped]
        summary_rows.append(
            {
                "device": key[0],
                "cipher": key[1],
                "operation": key[2],
                "block_size": key[3],
                "timing_scope": key[4],
                "count": str(len(grouped)),
                "time_ms_min": f"{min(times):.3f}",
                "time_ms_mean": f"{statistics.mean(times):.3f}",
                "time_ms_median": f"{statistics.median(times):.3f}",
                "time_ms_max": f"{max(times):.3f}",
                "gib_s_min": f"{min(throughputs):.3f}",
                "gib_s_mean": f"{statistics.mean(throughputs):.3f}",
                "gib_s_median": f"{statistics.median(throughputs):.3f}",
                "gib_s_max": f"{max(throughputs):.3f}",
            }
        )
    return summary_rows


def write_markdown(output: Path, summary_rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "device",
        "cipher",
        "operation",
        "block_size",
        "timing_scope",
        "count",
        "time_ms_min",
        "time_ms_mean",
        "time_ms_median",
        "time_ms_max",
        "gib_s_min",
        "gib_s_mean",
        "gib_s_median",
        "gib_s_max",
    ]
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# CUDA-AES Benchmark Summary\n\n")
        handle.write("Generated from raw Phase 3 benchmark CSV output.\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary_rows:
            handle.write("| " + " | ".join(row[column] for column in columns) + " |\n")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.inputs)
    summary_rows = summarize(rows)
    write_markdown(args.output, summary_rows)
    print(f"Wrote {len(summary_rows)} summary rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

