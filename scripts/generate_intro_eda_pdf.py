#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _wrap_text(s: str, width: int = 105) -> str:
    lines = []
    for block in s.split("\n"):
        if not block.strip():
            lines.append("")
            continue
        lines.extend(wrap(block, width=width))
    return "\n".join(lines)


def _add_page_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=18, weight="bold", va="top")
    if subtitle:
        ax.text(0.02, 0.91, subtitle, fontsize=11, va="top", color="#444444")


def _draw_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, footnote: str | None = None, max_rows: int = 26) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    _add_page_title(ax, title)
    view = df.head(max_rows).copy()
    ax_tbl = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    if footnote:
        ax.text(0.02, 0.03, footnote, fontsize=9, color="#444444")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(project_root: Path, out_pdf: Path) -> None:
    eda_dir = project_root / "outputs" / "eda"

    basic = json.loads((eda_dir / "basic_profile.json").read_text(encoding="utf-8"))
    col_summary = pd.read_csv(eda_dir / "column_summary.csv")
    event_counts = pd.read_csv(eda_dir / "event_type_counts.csv")
    coverage = pd.read_csv(eda_dir / "milestone_coverage_ordered_test.csv")
    timeline_all = pd.read_csv(eda_dir / "timeline_average_all.csv")
    timeline_filtered = pd.read_csv(eda_dir / "timeline_average_filter_test_code_CBCD.csv")
    likelihood = pd.read_csv(eda_dir / "event_likelihood_ab.csv")

    intro_text = (
        "Clinical laboratory testing follows a time-based specimen journey: a physician places a lab order, "
        "a specimen is collected, transported through the lab network, received by the performing lab, "
        "analyzed, and then verified results are returned to care teams. This process matters because delays "
        "or defects can directly impact treatment timing, lab efficiency, and patient safety.\n\n"
        "Stakeholders include clinical operations leaders, laboratory managers, and quality/safety teams. "
        "They need a visual tool to answer three decision questions: (1) what the average timeline looks like "
        "for a selected subgroup, (2) how two mutually exclusive cohorts differ (A/B), and (3) how likely a "
        "specific defect event is across cohorts.\n\n"
        "Our approach is to model the process at ordered-test grain using accession_id + test_code, compute "
        "milestone offsets relative to test_ordered_dt, run EDA for quality/distribution checks, and then design "
        "robust timeline and likelihood views based on medians and distribution spread rather than means alone."
    )

    eda_text = (
        f"The 2025 extract contains {basic['shape'][0]:,} rows and {basic['shape'][1]} columns, spanning "
        f"{basic['time_min']} to {basic['time_max']}. It includes {basic['n_accession_id']:,} unique orders "
        f"(accession_id), {basic['n_test_code']:,} test codes, {basic['n_event_type']} event types, and "
        f"{basic['n_event_source']} event sources (order, tube_tracker).\n\n"
        "Missingness is low for most fields; barcode has the largest null rate (~3.67%), while test_code is ~0.11%. "
        "Event counts show core ordered-test milestones dominate the extract (ordered, collected, receipt, "
        "resulted, verified). At ordered-test level, core milestone coverage remains high and cancellation appears as "
        "a meaningful defect subset.\n\n"
        "Distribution checks show long-tailed timing behavior, so medians and interquartile ranges are more reliable "
        "than means for communication. A filtered example (test_code = CBCD) reveals tighter timing than the global "
        "population, confirming the need for dimensional filtering. In A/B likelihood, cancellation differs by cohort "
        "(Hospital vs Medical), demonstrating why rate-normalized defect views are required in addition to timeline summaries."
    )

    with PdfPages(out_pdf) as pdf:
        # Page 1: title + sections
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        _add_page_title(
            ax,
            "SI 649 Report (Draft): Introduction + Exploratory Data Analysis",
            "Generated from local EDA outputs",
        )
        ax.text(0.02, 0.86, "Introduction", fontsize=14, weight="bold", va="top")
        ax.text(0.02, 0.84, _wrap_text(intro_text, 98), fontsize=10.5, va="top")
        ax.text(0.02, 0.47, "Exploratory Data Analysis", fontsize=14, weight="bold", va="top")
        ax.text(0.02, 0.45, _wrap_text(eda_text, 98), fontsize=10.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: basic profile + null rates
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis("off")
        ax1.set_title("Dataset Profile", loc="left", fontsize=13, weight="bold")
        profile_rows = [
            ["Rows", f"{basic['shape'][0]:,}"],
            ["Columns", f"{basic['shape'][1]}"],
            ["Time Start", basic["time_min"]],
            ["Time End", basic["time_max"]],
            ["Unique accession_id", f"{basic['n_accession_id']:,}"],
            ["Unique test_code", f"{basic['n_test_code']:,}"],
            ["Unique event_type", f"{basic['n_event_type']}"],
            ["Unique event_source", f"{basic['n_event_source']}"],
        ]
        tbl = ax1.table(cellText=profile_rows, colLabels=["Metric", "Value"], loc="upper left", cellLoc="left", colLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.4)

        ax2 = fig.add_subplot(gs[0, 1])
        null_df = col_summary.sort_values("null_rate", ascending=False).head(10)
        ax2.barh(null_df["column"], null_df["null_rate"] * 100, color="#4c78a8")
        ax2.invert_yaxis()
        ax2.set_title("Top Null Rates (%)", loc="left", fontsize=13, weight="bold")
        ax2.set_xlabel("Percent")
        for i, v in enumerate(null_df["null_rate"] * 100):
            ax2.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)

        ax3 = fig.add_subplot(gs[1, :])
        top_events = event_counts.head(14).copy()
        labels = top_events["event_type"] + " (" + top_events["event_source"] + ")"
        ax3.bar(labels, top_events["count"], color=["#72b7b2" if s == "order" else "#f58518" for s in top_events["event_source"]])
        ax3.set_title("Top Event Counts", loc="left", fontsize=13, weight="bold")
        ax3.set_ylabel("Count")
        ax3.tick_params(axis="x", labelrotation=35)
        ax3.text(0.01, 0.95, "Annotation: core ordered milestones dominate event volume.", transform=ax3.transAxes, fontsize=9, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: milestone coverage + timeline summary
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(1, 2, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        cov = coverage.copy()
        ax1.barh(cov["milestone"], cov["coverage"] * 100, color="#54a24b")
        ax1.invert_yaxis()
        ax1.set_title("Milestone Coverage (Ordered-Test Level)", loc="left", fontsize=13, weight="bold")
        ax1.set_xlabel("Coverage (%)")
        for i, v in enumerate(cov["coverage"] * 100):
            ax1.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)
        ax1.text(0.02, 0.04, "Annotation: Core milestones are highly complete; cancellation is a smaller defect subset.", transform=ax1.transAxes, fontsize=9)

        ax2 = fig.add_subplot(gs[0, 1])
        t = timeline_all.copy()
        y = range(len(t))
        ax2.errorbar(
            t["median_h"],
            y,
            xerr=[t["median_h"] - t["p25_h"], t["p75_h"] - t["median_h"]],
            fmt="o",
            capsize=3,
            color="#4c78a8",
        )
        ax2.set_yticks(list(y))
        ax2.set_yticklabels(t["milestone"])
        ax2.set_title("Global Timeline Offsets (Median with IQR)", loc="left", fontsize=13, weight="bold")
        ax2.set_xlabel("Hours since test_ordered_dt")
        ax2.grid(axis="x", alpha=0.3)
        ax2.text(0.02, 0.04, "Annotation: median is preferred due to long-tailed timing distribution.", transform=ax2.transAxes, fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: filtered timeline + defect likelihood
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(1, 2, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        all_t = timeline_all.set_index("milestone")
        fil_t = timeline_filtered.set_index("milestone")
        mlist = [m for m in all_t.index if m in fil_t.index]
        ax1.plot([all_t.loc[m, "median_h"] for m in mlist], mlist, marker="o", label="All")
        ax1.plot([fil_t.loc[m, "median_h"] for m in mlist], mlist, marker="o", label="Filter: test_code=CBCD")
        ax1.set_title("Average Timeline: Global vs Filtered", loc="left", fontsize=13, weight="bold")
        ax1.set_xlabel("Median hours since test_ordered_dt")
        ax1.grid(axis="x", alpha=0.3)
        ax1.legend()
        ax1.text(0.02, 0.04, "Annotation: filtering reveals subgroup process behavior hidden in global aggregates.", transform=ax1.transAxes, fontsize=9)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(likelihood["group"], likelihood["likelihood"] * 100, color=["#4c78a8", "#f58518"])
        ax2.set_title("Event Likelihood (Defect): cancellation_dt", loc="left", fontsize=13, weight="bold")
        ax2.set_ylabel("Likelihood (%)")
        for i, (_, row) in enumerate(likelihood.iterrows()):
            ax2.text(i, row["likelihood"] * 100, f"{row['likelihood']*100:.2f}%\n({int(row['defect_n'])}/{int(row['total_ordered_tests'])})", ha="center", va="bottom", fontsize=9)
        ax2.text(0.02, 0.04, "Annotation: A/B likelihood supports risk-focused operational comparison.", transform=ax2.transAxes, fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Table appendix pages
        _draw_table_page(
            pdf,
            "Appendix Table A: Column Summary",
            col_summary,
            footnote="Source: outputs/eda/column_summary.csv",
            max_rows=20,
        )
        _draw_table_page(
            pdf,
            "Appendix Table B: Event Type Counts (Top Rows)",
            event_counts,
            footnote="Source: outputs/eda/event_type_counts.csv",
            max_rows=30,
        )
        _draw_table_page(
            pdf,
            "Appendix Table C: Milestone Coverage",
            coverage,
            footnote="Source: outputs/eda/milestone_coverage_ordered_test.csv",
            max_rows=20,
        )
        _draw_table_page(
            pdf,
            "Appendix Table D: Timeline (All)",
            timeline_all,
            footnote="Source: outputs/eda/timeline_average_all.csv",
            max_rows=20,
        )
        _draw_table_page(
            pdf,
            "Appendix Table E: Timeline (Filter test_code=CBCD)",
            timeline_filtered,
            footnote="Source: outputs/eda/timeline_average_filter_test_code_CBCD.csv",
            max_rows=20,
        )
        _draw_table_page(
            pdf,
            "Appendix Table F: Event Likelihood A/B",
            likelihood,
            footnote="Source: outputs/eda/event_likelihood_ab.csv",
            max_rows=20,
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "report_intro_eda.pdf"
    build_pdf(root, out)
    print(f"Generated: {out}")
