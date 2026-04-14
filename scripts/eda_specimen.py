#!/usr/bin/env python
"""
EDA for SI649 specimen journey dataset.

Focus:
1) Average timeline view (+ dimensional filter example)
2) A/B comparison
3) Event likelihood for a defect event
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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


def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = s
    for c in bad:
        out = out.replace(c, "_")
    return out


def _summarize_series(x: pd.Series, label: str) -> dict:
    x = x.dropna()
    return {
        "milestone": label,
        "n": int(x.shape[0]),
        "mean_h": float(x.mean()) if len(x) else None,
        "median_h": float(x.median()) if len(x) else None,
        "p25_h": float(x.quantile(0.25)) if len(x) else None,
        "p75_h": float(x.quantile(0.75)) if len(x) else None,
        "p90_h": float(x.quantile(0.90)) if len(x) else None,
    }


def _plot_if_available(
    timeline_df: pd.DataFrame,
    ab_df: pd.DataFrame,
    likelihood_df: pd.DataFrame,
    outdir: Path,
    filter_title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipped PNG export.")
        return

    # 1) Average timeline
    t = timeline_df.copy()
    t["milestone"] = pd.Categorical(t["milestone"], categories=MILESTONES, ordered=True)
    t = t.sort_values("milestone")

    fig, ax = plt.subplots(figsize=(10, 4))
    y = t["milestone"].astype(str)
    ax.errorbar(
        t["median_h"],
        y,
        xerr=[t["median_h"] - t["p25_h"], t["p75_h"] - t["median_h"]],
        fmt="o",
        capsize=3,
    )
    ax.set_xlabel("Hours since test_ordered_dt")
    ax.set_ylabel("Milestone")
    ax.set_title(f"Average Timeline ({filter_title})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"timeline_{_safe_filename(filter_title)}.png", dpi=180)
    plt.close(fig)

    # 2) A/B comparison
    a = ab_df.copy()
    if a.empty:
        return
    a["milestone"] = pd.Categorical(a["milestone"], categories=MILESTONES, ordered=True)
    a = a.sort_values("milestone")
    groups = a["group"].dropna().unique().tolist()
    if len(groups) != 2:
        return
    g1, g2 = groups[0], groups[1]
    pvt = a.pivot(index="milestone", columns="group", values="median_h").dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    y = range(len(pvt))
    ax.scatter(pvt[g1], y, label=str(g1))
    ax.scatter(pvt[g2], y, label=str(g2))
    for i, (_, row) in enumerate(pvt.iterrows()):
        ax.plot([row[g1], row[g2]], [i, i], alpha=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(pvt.index.astype(str))
    ax.set_xlabel("Median hours since test_ordered_dt")
    ax.set_title("A/B Timeline Comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ab_timeline_comparison.png", dpi=180)
    plt.close(fig)

    # 3) Event likelihood
    l = likelihood_df.copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(l["group"].astype(str), l["likelihood"], color=["#4c78a8", "#f58518"][: len(l)])
    ax.set_ylim(0, max(0.01, l["likelihood"].max() * 1.25))
    ax.set_ylabel("Likelihood")
    ax.set_title(f"Event Likelihood: {l['defect_event'].iloc[0]}")
    for i, (_, row) in enumerate(l.iterrows()):
        ax.text(i, row["likelihood"], f"{row['likelihood']:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "event_likelihood_ab.png", dpi=180)
    plt.close(fig)


def _timeline_summary(ordered_wide: pd.DataFrame, key_filter: pd.Series | None = None) -> pd.DataFrame:
    sub = ordered_wide if key_filter is None else ordered_wide.loc[key_filter].copy()
    rows = []
    for m in MILESTONES:
        if m not in sub.columns:
            continue
        rows.append(_summarize_series(sub[f"offset_{m}_h"], m))
    return pd.DataFrame(rows)


def _first_non_null_per_key(df: pd.DataFrame, key: str, cols: Iterable[str]) -> pd.DataFrame:
    work = df[[key, *cols, "event_dt"]].sort_values("event_dt")
    return work.drop_duplicates(subset=[key], keep="first").drop(columns=["event_dt"])


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
    df = pd.read_csv(in_path, sep="\t", dtype=dtypes)
    df["event_dt"] = pd.to_datetime(df["event_dt"], errors="coerce", utc=True)
    if args.year is not None:
        df = df.loc[df["event_dt"].dt.year == args.year].copy()

    # Basic profile
    basic = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "time_min": str(df["event_dt"].min()),
        "time_max": str(df["event_dt"].max()),
        "n_accession_id": int(df["accession_id"].nunique()),
        "n_test_code": int(df["test_code"].nunique(dropna=True)),
        "n_event_type": int(df["event_type"].nunique()),
        "n_event_source": int(df["event_source"].nunique()),
    }
    (outdir / "basic_profile.json").write_text(json.dumps(basic, indent=2), encoding="utf-8")

    col_summary = pd.DataFrame(
        {
            "column": df.columns,
            "null_count": [int(df[c].isna().sum()) for c in df.columns],
            "null_rate": [float(df[c].isna().mean()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    col_summary.to_csv(outdir / "column_summary.csv", index=False)

    df["event_source"] = df["event_source"].astype("string")
    df["event_type"] = df["event_type"].astype("string")
    top_event_types = (
        df.groupby(["event_source", "event_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_event_types.to_csv(outdir / "event_type_counts.csv", index=False)

    # Ordered-test level EDA key: accession_id + test_code
    order_df = df.loc[(df["event_source"] == "order") & (df["test_code"].notna())].copy()
    order_df["ot_key"] = order_df["accession_id"] + "||" + order_df["test_code"]

    # First timestamp per key/event_type
    first_events = (
        order_df.sort_values("event_dt")
        .drop_duplicates(subset=["ot_key", "event_type"], keep="first")
        .loc[:, ["ot_key", "event_type", "event_dt"]]
    )
    wide = first_events.pivot(index="ot_key", columns="event_type", values="event_dt").reset_index()

    # Keep dimensions from ordered moment
    ordered_anchor = order_df.loc[order_df["event_type"] == "test_ordered_dt"].copy()
    key_dims = _first_non_null_per_key(
        ordered_anchor,
        "ot_key",
        [
            "accession_id",
            "test_code",
            "pat_enc_csn_id",
            "pat_mrn_id",
            "event_street",
            "test_performing_dept",
            "test_performing_location",
        ],
    )
    ordered_wide = key_dims.merge(wide, on="ot_key", how="left")

    # Offsets relative to test_ordered_dt
    for m in MILESTONES:
        if m in ordered_wide.columns:
            ordered_wide[f"offset_{m}_h"] = (
                (ordered_wide[m] - ordered_wide["test_ordered_dt"]).dt.total_seconds() / 3600.0
            )

    # Milestone coverage at ordered-test grain
    n_tests = int(ordered_wide.shape[0])
    coverage_rows = []
    for m in MILESTONES:
        if m in ordered_wide.columns:
            n = int(ordered_wide[m].notna().sum())
            coverage_rows.append({"milestone": m, "n": n, "coverage": n / n_tests if n_tests else None})
    pd.DataFrame(coverage_rows).to_csv(outdir / "milestone_coverage_ordered_test.csv", index=False)

    # Average timeline (all)
    timeline_all = _timeline_summary(ordered_wide)
    timeline_all.to_csv(outdir / "timeline_average_all.csv", index=False)

    # Average timeline (filtered example)
    # Example filter: event_street == args.filter_value on args.filter_column
    if args.filter_column in ordered_wide.columns:
        mask = ordered_wide[args.filter_column].astype("string") == pd.Series(
            [args.filter_value] * len(ordered_wide), index=ordered_wide.index, dtype="string"
        )
        timeline_filtered = _timeline_summary(ordered_wide, mask)
    else:
        timeline_filtered = pd.DataFrame()
    timeline_filtered.to_csv(
        outdir / f"timeline_average_filter_{_safe_filename(args.filter_column)}_{_safe_filename(args.filter_value)}.csv",
        index=False,
    )

    # A/B comparison
    ab = ordered_wide.loc[ordered_wide[args.ab_group_col].isin([args.group_a, args.group_b])].copy()
    ab_rows = []
    for g, gdf in ab.groupby(args.ab_group_col):
        for m in MILESTONES:
            off_col = f"offset_{m}_h"
            if off_col in gdf.columns:
                rec = _summarize_series(gdf[off_col], m)
                rec["group"] = g
                ab_rows.append(rec)
    ab_summary = pd.DataFrame(ab_rows)
    ab_summary.to_csv(outdir / "ab_timeline_summary.csv", index=False)

    # Defect likelihood in A/B (default: cancellation_dt)
    defect_mask = ordered_wide[args.defect_event].notna() if args.defect_event in ordered_wide.columns else pd.Series(
        [False] * len(ordered_wide), index=ordered_wide.index
    )
    likelihood_rows = []
    for g, gdf in ab.groupby(args.ab_group_col):
        total = int(gdf.shape[0])
        defect_n = int(defect_mask.loc[gdf.index].sum())
        likelihood_rows.append(
            {
                "group": g,
                "defect_event": args.defect_event,
                "total_ordered_tests": total,
                "defect_n": defect_n,
                "likelihood": (defect_n / total) if total else None,
            }
        )
    likelihood_df = pd.DataFrame(likelihood_rows)
    likelihood_df.to_csv(outdir / "event_likelihood_ab.csv", index=False)

    # Optional plot export
    plot_filter_title = f"{args.filter_column}={args.filter_value}"
    _plot_if_available(
        timeline_df=timeline_filtered if not timeline_filtered.empty else timeline_all,
        ab_df=ab_summary,
        likelihood_df=likelihood_df,
        outdir=outdir,
        filter_title=plot_filter_title,
    )

    # Save the modeling table for downstream visualization code
    # Prefer parquet; fallback to CSV when parquet engine is unavailable.
    try:
        ordered_wide.to_parquet(outdir / "ordered_test_level_table.parquet", index=False)
    except Exception:
        ordered_wide.to_csv(outdir / "ordered_test_level_table.csv", index=False)

    print(f"EDA complete. Output directory: {outdir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EDA for SI649 specimen journey dataset")
    p.add_argument(
        "--input",
        type=str,
        default="2025_specimen_time_series_events_no_phi.tsv",
        help="Path to input TSV file",
    )
    p.add_argument("--outdir", type=str, default="outputs/eda", help="Directory for EDA outputs")
    p.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Filter event_dt by year; set empty by passing --year 0",
    )
    p.add_argument(
        "--filter-column",
        type=str,
        default="test_code",
        help="Dimensional filter column for timeline example",
    )
    p.add_argument(
        "--filter-value",
        type=str,
        default="CBCD",
        help="Dimensional filter value for timeline example",
    )
    p.add_argument(
        "--ab-group-col",
        type=str,
        default="event_street",
        help="Column used for A/B grouping",
    )
    p.add_argument("--group-a", type=str, default="Hospital", help="A group value")
    p.add_argument("--group-b", type=str, default="Medical", help="B group value")
    p.add_argument(
        "--defect-event",
        type=str,
        default="cancellation_dt",
        help="Defect event column in ordered-test table (e.g., cancellation_dt)",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.year == 0:
        args.year = None
    run(args)
