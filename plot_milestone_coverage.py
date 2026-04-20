#!/usr/bin/env python3
"""
Generate EDA figure: milestone coverage bar chart (for report §3.4.3).

Reads:  outputs/eda/milestone_coverage_ordered_test.csv  (from scripts/eda_specimen.py)
Writes: outputs/eda/figure_milestone_coverage.png

This script lives under scripts/ so it can be committed; outputs/ is gitignored.

Run from repo root:
  python scripts/plot_milestone_coverage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_figure(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    milestones = [
        "test_ordered_dt",
        "test_collected_dt",
        "test_receipt_dt",
        "test_min_resulted_dt",
        "test_min_verified_dt",
        "test_max_resulted_dt",
        "test_max_verified_dt",
        "cancellation_dt",
    ]
    labels = {
        "test_ordered_dt": "Order Placed",
        "test_collected_dt": "Collected",
        "test_receipt_dt": "Lab Received",
        "test_min_resulted_dt": "First Result",
        "test_min_verified_dt": "First Verified",
        "test_max_resulted_dt": "Last Result",
        "test_max_verified_dt": "Final Verified",
        "cancellation_dt": "Cancelled",
    }

    present = [m for m in milestones if m in set(df["milestone"])]
    df = df.set_index("milestone").reindex(present).reset_index()
    df["label"] = df["milestone"].map(labels).fillna(df["milestone"])

    n_den = df.loc[df["milestone"].eq("test_ordered_dt"), "n"]
    n_total = int(n_den.iloc[0]) if len(n_den) else int(df["n"].max())

    vals = (df["coverage"].astype(float).values * 100.0).tolist()
    colors = ["#4C78A8"] * len(df)
    if "cancellation_dt" in set(df["milestone"]):
        idx = int(df.index[df["milestone"].eq("cancellation_dt")][0])
        colors[idx] = "#F58518"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(range(len(df)), vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Coverage (% of ordered tests)")
    ax.set_title(f"Milestone coverage at ordered-test grain (N={n_total:,})")
    ax.set_xticks(list(range(len(df))))
    ax.set_xticklabels(df["label"], rotation=25, ha="right")

    for i, v in enumerate(vals):
        ax.text(i, min(103.0, v + 1.2), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    default_csv = root / "outputs" / "eda" / "milestone_coverage_ordered_test.csv"
    default_png = root / "outputs" / "eda" / "figure_milestone_coverage.png"

    p = argparse.ArgumentParser(description="Plot milestone coverage bar chart for report.")
    p.add_argument("--input", type=Path, default=default_csv, help="Path to milestone_coverage_ordered_test.csv")
    p.add_argument("--output", type=Path, default=default_png, help="Path to output PNG")
    args = p.parse_args()

    build_figure(args.input, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
