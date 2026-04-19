#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MILESTONES = [
    "test_ordered_dt",
    "test_collected_dt",
    "test_receipt_dt",
    "test_min_resulted_dt",
    "test_min_verified_dt",
    "test_max_resulted_dt",
    "test_max_verified_dt",
    "cancellation_dt",
]


def build_ordered_test_table(input_tsv: Path, year: int | None = None) -> pd.DataFrame:
    dtypes = {
        "accession_id": "string",
        "pat_enc_csn_id": "string",
        "pat_mrn_id": "string",
        "barcode": "string",
        "tube_id": "string",
        "specimen_id": "string",
        "test_code": "string",
        "test_performing_dept": "string",
        "test_performing_location": "string",
        "event_street": "string",
        "event_source": "string",
        "event_type": "string",
        "event_dt": "string",
    }
    df = pd.read_csv(input_tsv, sep="\t", dtype=dtypes)
    df["event_dt"] = pd.to_datetime(df["event_dt"], errors="coerce", utc=True)

    if year is not None:
        df = df[df["event_dt"].dt.year == year].copy()

    order_df = df[(df["event_source"] == "order") & (df["test_code"].notna())].copy()
    order_df["ot_key"] = order_df["accession_id"] + "||" + order_df["test_code"]

    # Keep only milestone events needed by the dashboard/report.
    order_df = order_df[order_df["event_type"].isin(MILESTONES)].copy()

    # Earliest timestamp per ordered-test key and event type.
    first_events = (
        order_df.sort_values("event_dt")
        .drop_duplicates(subset=["ot_key", "event_type"], keep="first")
        .loc[:, ["ot_key", "event_type", "event_dt"]]
    )
    wide = first_events.pivot(index="ot_key", columns="event_type", values="event_dt").reset_index()

    # Use ordered event rows as anchor for dimensions.
    ordered_anchor = order_df[order_df["event_type"] == "test_ordered_dt"].copy()
    key_dims = (
        ordered_anchor.sort_values("event_dt")
        .drop_duplicates(subset=["ot_key"], keep="first")
        .loc[
            :,
            [
                "ot_key",
                "accession_id",
                "test_code",
                "pat_enc_csn_id",
                "pat_mrn_id",
                "event_street",
                "test_performing_dept",
                "test_performing_location",
            ],
        ]
    )
    ordered_wide = key_dims.merge(wide, on="ot_key", how="left")

    # Offset hours from test_ordered_dt.
    for m in MILESTONES:
        if m in ordered_wide.columns:
            ordered_wide[f"offset_{m}_h"] = (
                (ordered_wide[m] - ordered_wide["test_ordered_dt"]).dt.total_seconds() / 3600.0
            )

    return ordered_wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ordered-test-level table for dashboard input."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("2025_specimen_time_series_events_no_phi.tsv"),
        help="Path to raw TSV input file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/eda"),
        help="Output directory for ordered_test_level_table.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=0,
        help="Filter event_dt by year. Default 0 means no year filter (use all data).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    year = None if args.year == 0 else args.year
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ordered_wide = build_ordered_test_table(args.input, year=year)

    parquet_path = outdir / "ordered_test_level_table.parquet"
    csv_path = outdir / "ordered_test_level_table.csv"
    try:
        ordered_wide.to_parquet(parquet_path, index=False)
        print(f"Wrote: {parquet_path}")
    except Exception:
        ordered_wide.to_csv(csv_path, index=False)
        print(f"Parquet write failed; wrote CSV instead: {csv_path}")

    print(f"Rows: {len(ordered_wide):,}")


if __name__ == "__main__":
    main()
