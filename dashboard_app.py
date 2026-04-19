from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, norm
import streamlit as st


st.set_page_config(page_title="Specimen Journey Dashboard", layout="wide")

DATA_CANDIDATES = [
    Path("outputs/eda/ordered_test_level_table.parquet"),
    Path("outputs/eda/ordered_test_level_table.csv"),
    Path("data/ordered_test_level_table.parquet"),
    Path("data/ordered_test_level_table.csv"),
]
RAW_TSV_FILE = Path("2025_specimen_time_series_events_no_phi.tsv")
RAW_PARTS_GLOB = "data/raw_parts/2025_specimen_time_series_events_no_phi.part*.tsv"

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

MEDIAN_BASELINE_LABEL = "All Others (Complement)"

GROUPABLE_COLS = [
    "order_day_type",
    "event_street",
    "test_performing_dept",
    "test_performing_location",
    "test_code",
]

GROUPABLE_LABELS = {
    "order_day_type": "Order Day Type (Weekday/Weekend)",
    "event_street": "Collection/Transit Street",
    "test_performing_dept": "Performing Lab Department",
    "test_performing_location": "Performing Lab Location",
    "test_code": "Ordered Test Code",
}

MILESTONE_LABELS = {
    "test_ordered_dt": "Order Placed",
    "test_collected_dt": "Specimen Collected",
    "test_receipt_dt": "Lab Receipt",
    "test_min_resulted_dt": "First Result Produced",
    "test_min_verified_dt": "First Result Verified",
    "test_max_resulted_dt": "Last Result Produced",
    "test_max_verified_dt": "Last Result Verified",
    "cancellation_dt": "Order/Test Cancelled",
}

@st.cache_data(show_spinner=True)
def _build_ordered_test_table_from_raw_parts(parts: list[Path]) -> pd.DataFrame:
    usecols = [
        "accession_id",
        "test_code",
        "event_street",
        "test_performing_dept",
        "test_performing_location",
        "event_source",
        "event_type",
        "event_dt",
    ]
    dtypes = {c: "string" for c in usecols}
    frames = []
    for p in sorted(parts):
        frames.append(pd.read_csv(p, sep="\t", usecols=usecols, dtype=dtypes))

    raw = pd.concat(frames, ignore_index=True)
    raw["event_dt"] = pd.to_datetime(raw["event_dt"], errors="coerce", utc=True)

    order = raw[(raw["event_source"] == "order") & (raw["test_code"].notna())].copy()
    order["ot_key"] = order["accession_id"] + "||" + order["test_code"]

    # Keep only milestones used by dashboard.
    needed = [
        "test_ordered_dt",
        "test_collected_dt",
        "test_receipt_dt",
        "test_min_resulted_dt",
        "test_max_verified_dt",
        "cancellation_dt",
    ]
    order = order[order["event_type"].isin(needed)].copy()

    # First timestamp per ordered-test per event type.
    first = (
        order.sort_values("event_dt")
        .drop_duplicates(subset=["ot_key", "event_type"], keep="first")
        .pivot(index="ot_key", columns="event_type", values="event_dt")
        .reset_index()
    )

    # Anchor dimensions from ordered event rows.
    dims = order[order["event_type"] == "test_ordered_dt"].copy()
    dims = dims.sort_values("event_dt").drop_duplicates(subset=["ot_key"], keep="first")
    dims = dims[
        ["ot_key", "accession_id", "test_code", "event_street", "test_performing_dept", "test_performing_location"]
    ]

    df = dims.merge(first, on="ot_key", how="left")

    # Compute offsets.
    all_m = [
        "test_ordered_dt",
        "test_collected_dt",
        "test_receipt_dt",
        "test_min_resulted_dt",
        "test_max_verified_dt",
        "cancellation_dt",
    ]
    for m in all_m:
        if m in df.columns:
            df[f"offset_{m}_h"] = (df[m] - df["test_ordered_dt"]).dt.total_seconds() / 3600.0
    return df


@st.cache_data(show_spinner=True)
def load_data() -> tuple[pd.DataFrame, str]:
    df = None
    source = ""
    for p in DATA_CANDIDATES:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            source = str(p)
            break

    if df is None:
        if RAW_TSV_FILE.exists():
            df = _build_ordered_test_table_from_raw_parts([RAW_TSV_FILE])
            source = str(RAW_TSV_FILE)
        else:
            parts = [Path(x) for x in sorted(Path(".").glob(RAW_PARTS_GLOB))]
            if not parts:
                raise FileNotFoundError(
                    "No dashboard data found. Expected one of: "
                    + ", ".join(str(x) for x in DATA_CANDIDATES)
                    + f", raw file {RAW_TSV_FILE}, or raw split parts matching {RAW_PARTS_GLOB}"
                )
            df = _build_ordered_test_table_from_raw_parts(parts)
            source = f"built from raw parts ({len(parts)} files)"

    dt_cols = [c for c in df.columns if c.endswith("_dt")]
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    # Derived A/B dimension: weekday vs weekend by order timestamp.
    wd = df["test_ordered_dt"].dt.weekday
    df["order_day_type"] = wd.map(lambda x: "Weekend" if pd.notna(x) and int(x) >= 5 else "Weekday")

    return df, source


def summarize_offsets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MILESTONES:
        col = f"offset_{m}_h"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        rows.append(
            {
                "milestone": m,
                "n": int(s.shape[0]),
                "median_h": s.median() if len(s) else None,
                "p25_h": s.quantile(0.25) if len(s) else None,
                "p75_h": s.quantile(0.75) if len(s) else None,
                "mean_h": s.mean() if len(s) else None,
            }
        )
    out = pd.DataFrame(rows)
    out["milestone"] = pd.Categorical(out["milestone"], categories=MILESTONES, ordered=True)
    return out.sort_values("milestone")


def make_timeline_fig(t: pd.DataFrame, title: str) -> go.Figure:
    t = t.copy()
    t["milestone_label"] = t["milestone"].map(MILESTONE_LABELS).fillna(t["milestone"])
    t = t[t["median_h"].notna()].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t["median_h"],
            y=t["milestone_label"],
            mode="markers+lines",
            name="Median",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(t["p75_h"] - t["median_h"]).fillna(0),
                arrayminus=(t["median_h"] - t["p25_h"]).fillna(0),
                thickness=1,
                width=4,
            ),
            marker=dict(size=9),
        )
    )
    milestone_order = [MILESTONE_LABELS.get(m, m) for m in MILESTONES]
    fig.update_layout(
        title=title,
        xaxis_title="Hours Since Order Placed",
        yaxis_title="Milestone",
        height=430,
        yaxis=dict(
            categoryorder="array",
            categoryarray=milestone_order,
        ),
    )
    return fig


def make_completion_curves_fig(df: pd.DataFrame, title: str) -> go.Figure:
    stage_cols = {
        "Collected": "offset_test_collected_dt_h",
        "Received": "offset_test_receipt_dt_h",
        "First Result": "offset_test_min_resulted_dt_h",
        "Final Verified": "offset_test_max_verified_dt_h",
    }

    # Build x-axis range from observed non-negative offsets.
    all_offsets = []
    for col in stage_cols.values():
        if col in df.columns:
            s = df[col].dropna()
            s = s[s >= 0]
            if len(s):
                all_offsets.append(s)
    if not all_offsets:
        return go.Figure()

    merged = pd.concat(all_offsets, ignore_index=True)
    _ = merged  # keep for possible future diagnostics
    x_max = 15.0
    x_grid = np.linspace(0, x_max, 120)

    fig = go.Figure()
    for label, col in stage_cols.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        s = s[s >= 0]
        if len(s) == 0:
            continue

        # Fraction completed by time t among rows that have this milestone timestamp.
        y = [(s <= t).mean() * 100 for t in x_grid]
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y,
                mode="lines",
                name=label,
                hovertemplate="Time=%{x:.2f}h<br>Completed=%{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hours Since Order Placed",
        xaxis=dict(range=[0, 15]),
        yaxis_title="Completed Samples (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=430,
        legend_title="Stage",
    )
    return fig


def make_ab_fig(df: pd.DataFrame, group_col: str, group_a: str, group_b: str) -> go.Figure:
    def _subset_for_group(g: str, other: str) -> pd.DataFrame:
        if g == MEDIAN_BASELINE_LABEL:
            if other == MEDIAN_BASELINE_LABEL:
                return df.iloc[0:0].copy()
            return df[df[group_col].notna() & (df[group_col].astype(str) != str(other))].copy()
        return df[df[group_col].astype(str) == str(g)].copy()

    rows = []
    for g in [group_a, group_b]:
        other = group_b if g == group_a else group_a
        sub = _subset_for_group(g, other)
        if sub.empty:
            continue
        s = summarize_offsets(sub)
        s["group"] = g
        rows.append(s)
    if not rows:
        return go.Figure()

    ab = pd.concat(rows, ignore_index=True)
    # Remove the baseline milestone in A/B view because it is always ~0 by definition.
    ab = ab[ab["milestone"] != "test_ordered_dt"].copy()
    ab["milestone_label"] = ab["milestone"].map(MILESTONE_LABELS).fillna(ab["milestone"])

    dim_label = GROUPABLE_LABELS.get(group_col, group_col)
    if group_col == "order_day_type":
        title = f"A/B Comparison: {dim_label}"
    else:
        title = f"A/B Comparison: {dim_label} ({group_a} vs {group_b})"

    fig = px.bar(
        ab,
        x="median_h",
        y="milestone_label",
        color="group",
        barmode="group",
        orientation="h",
        hover_data=["n", "p25_h", "p75_h", "mean_h"],
        title=title,
    )
    fig.update_layout(
        height=430,
        xaxis_title="Median Hours Since Order Placed",
        yaxis_title="Milestone",
        legend_title="Group",
    )
    return fig


def compute_ab_completion_stats(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    metric_col: str = "offset_test_max_verified_dt_h",
) -> dict:
    def _series_for_group(g: str, other: str) -> pd.Series:
        if g == MEDIAN_BASELINE_LABEL:
            if other == MEDIAN_BASELINE_LABEL:
                return pd.Series(dtype=float)
            mask = df[group_col].notna() & (df[group_col].astype(str) != str(other))
            s = df.loc[mask, metric_col].dropna().astype(float)
        else:
            s = df.loc[df[group_col].astype(str) == str(g), metric_col].dropna().astype(float)
        return s[s >= 0]

    a = _series_for_group(group_a, group_b)
    b = _series_for_group(group_b, group_a)

    if group_a == MEDIAN_BASELINE_LABEL and group_b == MEDIAN_BASELINE_LABEL:
        return {"ok": False, "reason": "Please choose at least one real group (not both All Others baseline)."}

    if len(a) < 2 or len(b) < 2:
        return {"ok": False, "reason": "Not enough data points for statistical testing."}

    test = mannwhitneyu(a, b, alternative="two-sided")
    med_a, med_b = float(np.median(a)), float(np.median(b))
    diff = med_a - med_b
    slower = group_a if diff > 0 else group_b
    faster = group_b if diff > 0 else group_a

    if test.pvalue < 0.05:
        sentence = (
            f"{slower} is significantly slower than {faster} in Order -> Final Verified timing "
            f"(median difference {abs(diff):.2f}h, p={test.pvalue:.3g})."
        )
    else:
        sentence = (
            f"No statistically significant difference between {group_a} and {group_b} "
            f"in Order -> Final Verified timing (median difference {abs(diff):.2f}h, p={test.pvalue:.3g})."
        )

    return {
        "ok": True,
        "pvalue": float(test.pvalue),
        "sentence": sentence,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def defect_rate(df: pd.DataFrame) -> pd.Series:
    return df["cancellation_dt"].notna()


def make_likelihood_fig(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
) -> tuple[go.Figure, pd.DataFrame]:
    if group_a == MEDIAN_BASELINE_LABEL and group_b == MEDIAN_BASELINE_LABEL:
        return go.Figure(), pd.DataFrame()

    rows = []
    for g in [group_a, group_b]:
        if g == MEDIAN_BASELINE_LABEL:
            other = group_b if g == group_a else group_a
            sub = df[df[group_col].notna() & (df[group_col].astype(str) != str(other))].copy()
        else:
            sub = df[df[group_col].astype(str) == str(g)].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "group": g,
                "defect_n": int(sub["cancellation_dt"].notna().sum()),
                "total_n": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return go.Figure(), out

    out["likelihood"] = out["defect_n"] / out["total_n"]
    # 95% CI (normal approximation) for a binomial proportion.
    out["se"] = (out["likelihood"] * (1 - out["likelihood"]) / out["total_n"]).pow(0.5)
    out["ci_low"] = (out["likelihood"] - 1.96 * out["se"]).clip(lower=0)
    out["ci_high"] = (out["likelihood"] + 1.96 * out["se"]).clip(upper=1)


    # Dot plot with confidence intervals is more readable for 2-group A/B.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=out["group"],
            y=out["likelihood"],
            mode="markers",
            marker=dict(size=12),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(out["ci_high"] - out["likelihood"]).fillna(0),
                arrayminus=(out["likelihood"] - out["ci_low"]).fillna(0),
                thickness=1.2,
                width=6,
            ),
            customdata=out[["defect_n", "total_n"]],
            hovertemplate=(
                "Group=%{x}<br>"
                "Likelihood=%{y:.2%}<br>"
                "Cancellation Count=%{customdata[0]}<br>"
                "Total=%{customdata[1]}<extra></extra>"
            ),
            name="Cancellation Likelihood",
        )
    )
    if len(out) == 2:
        fig.add_trace(
            go.Scatter(
                x=out["group"],
                y=out["likelihood"],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.4)", dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    y_min = max(0.0, float(out["ci_low"].min()) - 0.002)
    y_max = min(1.0, float(out["ci_high"].max()) + 0.002)
    span = max(0.0001, y_max - y_min)
    dtick = span / 5.0
    # Put labels beside the point estimates (left/right) to avoid overlapping error bars.
    for i, (_, row) in enumerate(out.iterrows()):
        x_shift = -16 if i % 2 == 0 else 16
        x_anchor = "right" if i % 2 == 0 else "left"
        fig.add_annotation(
            x=row["group"],
            y=float(row["likelihood"]),
            text=f"{float(row['likelihood']):.1%}",
            showarrow=False,
            xshift=x_shift,
            xanchor=x_anchor,
            yanchor="middle",
            font=dict(size=14),
        )

    fig.update_layout(
        title="Cancellation Likelihood",
        height=430,
        xaxis_title="Group",
        yaxis_title="Likelihood",
        yaxis=dict(
            tickformat=".1%",
            range=[y_min, y_max],
            dtick=dtick,
        ),
    )
    return fig, out


def compute_attainment_6h_table(
    df: pd.DataFrame, group_col: str, threshold_h: float = 6.0, top_n: int = 10
) -> pd.DataFrame:
    stage_map = {
        "Collected": "offset_test_collected_dt_h",
        "Received": "offset_test_receipt_dt_h",
        "First Result": "offset_test_min_resulted_dt_h",
        "Final Verified": "offset_test_max_verified_dt_h",
    }
    valid = df[df[group_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    top_groups = valid[group_col].value_counts().head(top_n)
    rows = []
    for g, n in top_groups.items():
        sub = valid[valid[group_col] == g]
        rec = {"group": str(g), "n": int(n)}
        for stage_name, col in stage_map.items():
            if col in sub.columns:
                s = sub[col].dropna()
                rate = float((s <= threshold_h).mean()) if len(s) else np.nan
            else:
                rate = np.nan
            rec[stage_name] = rate
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["group_label"] = out["group"] + " (" + out["n"].map(lambda x: f"{x:,}") + ")"
    return out


def make_attainment_6h_heatmap(
    df: pd.DataFrame, group_col: str, threshold_h: float = 6.0, top_n: int = 10
) -> tuple[go.Figure, pd.DataFrame]:
    t = compute_attainment_6h_table(df, group_col, threshold_h=threshold_h, top_n=top_n)
    if t.empty:
        return go.Figure(), t

    threshold_label = _format_duration_label(threshold_h)
    stage_cols = ["Collected", "Received", "First Result", "Final Verified"]
    z = t[stage_cols].to_numpy() * 100.0
    y = t["group_label"].tolist()
    text = [[f"{v:.1f}%" if pd.notna(v) else "N/A" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=stage_cols,
            y=y,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title=f"Within {threshold_label} (%)"),
            hovertemplate=f"Group=%{{y}}<br>Stage=%{{x}}<br>{threshold_label} Attainment=%{{z:.1f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"{threshold_label} attainment by {GROUPABLE_LABELS.get(group_col, group_col)} "
            f"(Top {min(top_n, len(t))} by volume)"
        ),
        xaxis_title="Stage",
        yaxis=dict(
            title="Group (sample size)",
            autorange="reversed",
        ),
        height=450,
    )
    return fig, t


def compute_likelihood_pvalue(out: pd.DataFrame, group_a: str, group_b: str) -> dict:
    if out.shape[0] != 2:
        return {"ok": False, "reason": "Likelihood test currently supports exactly 2 groups."}

    a = out.loc[out["group"] == group_a]
    b = out.loc[out["group"] == group_b]
    if a.empty or b.empty:
        return {"ok": False, "reason": "Could not find both selected groups in likelihood output."}

    a_def, a_n = int(a["defect_n"].iloc[0]), int(a["total_n"].iloc[0])
    b_def, b_n = int(b["defect_n"].iloc[0]), int(b["total_n"].iloc[0])
    if a_n == 0 or b_n == 0:
        return {"ok": False, "reason": "One of the groups has zero samples."}

    p1 = a_def / a_n
    p2 = b_def / b_n
    p_pool = (a_def + b_def) / (a_n + b_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / a_n + 1 / b_n))
    if se == 0:
        return {"ok": False, "reason": "Unable to compute standard error (se=0)."}

    z = (p1 - p2) / se
    pval = 2 * norm.sf(abs(z))
    higher = group_a if p1 > p2 else group_b
    lower = group_b if p1 > p2 else group_a
    abs_pp = abs(p1 - p2) * 100

    if pval < 0.05:
        sentence = (
            f"{higher} has a significantly higher defect likelihood than {lower} "
            f"({abs_pp:.2f} percentage points, p={pval:.3g})."
        )
    else:
        sentence = (
            f"No statistically significant defect-likelihood difference between {group_a} and {group_b} "
            f"({abs_pp:.2f} percentage points, p={pval:.3g})."
        )

    return {"ok": True, "pvalue": float(pval), "z": float(z), "sentence": sentence}


def remove_negative_offset_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Remove rows only when any available offset column is negative.
    existing = [c for c in df.columns if c.startswith("offset_") and c.endswith("_h")]
    if not existing:
        return df, 0

    bad_mask = pd.Series(False, index=df.index)
    for c in existing:
        bad_mask = bad_mask | df[c].lt(0).fillna(False)

    removed = int(bad_mask.sum())
    cleaned = df.loc[~bad_mask].copy()
    return cleaned, removed


def _group_values_for_col(df: pd.DataFrame, group_col: str) -> list[str]:
    vals = sorted(df[group_col].dropna().astype(str).unique().tolist())
    # "All Others" is meaningful only if there are at least two concrete groups.
    if len(vals) >= 2:
        vals.append(MEDIAN_BASELINE_LABEL)
    return vals


def _render_dynamic_multiselect(
    label: str,
    key: str,
    option_df: pd.DataFrame,
    col: str,
    top_n: int,
) -> list[str]:
    options = option_df[col].dropna().astype(str).value_counts().head(top_n).index.tolist()
    existing = st.session_state.get(key, [])
    if not isinstance(existing, list):
        existing = []
    existing = [x for x in existing if x in options]
    st.session_state[key] = existing
    return st.multiselect(label, options=options, default=existing, key=key)


def _compute_dynamic_threshold_slider_settings(df: pd.DataFrame) -> tuple[float, float, float]:
    col = "offset_test_max_verified_dt_h"
    if col not in df.columns:
        return 0.5, 24.0, 6.0

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[s >= 0]
    if s.empty:
        return 0.5, 24.0, 6.0

    p95 = float(s.quantile(0.95))
    dynamic_max = p95 * 1.10

    # Dynamic slider granularity based on max horizon.
    if dynamic_max <= 2.0:
        step = 1.0 / 60.0  # 1 minute
    elif dynamic_max <= 12.0:
        step = 5.0 / 60.0  # 5 minutes
    else:
        step = 15.0 / 60.0  # 15 minutes

    # Keep a practical minimum upper bound and snap to step.
    slider_max = max(step, dynamic_max)
    slider_max = float(np.ceil(slider_max / step) * step)

    default_value = min(6.0, slider_max)
    default_value = float(np.round(default_value / step) * step)
    default_value = min(max(step, default_value), slider_max)

    return step, slider_max, default_value


def _format_duration_label(hours: float) -> str:
    total_min = int(round(hours * 60.0))
    if total_min < 120:
        return f"{total_min} min"
    h, m = divmod(total_min, 60)
    if m == 0:
        return f"{h} h"
    return f"{h}h {m}m"


def main() -> None:
    st.title("Specimen Journey Dashboard (V1)")
    st.caption("Interactive dashboard for average timeline, A/B comparison, and defect-event likelihood.")

    try:
        df, source = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unable to load dashboard data: {e}")
        st.stop()

    st.caption(f"Data source: {source}")

    with st.expander("Data Dictionary (What Each Variable Means)", expanded=False):
        dict_df = pd.DataFrame(
            [
                ["accession_id", "Lab order ID (one order can include multiple test codes)."],
                ["test_code", "Ordered test code; analysis grain uses accession_id + test_code."],
                ["event_street", "Street-level collection/transit location tied to the event."],
                ["test_performing_dept", "Lab department responsible for running the test."],
                ["test_performing_location", "Physical location of the performing lab."],
                ["test_ordered_dt", "Timestamp when the order was placed."],
                ["cancellation_dt", "Timestamp when the ordered test was cancelled (if applicable)."],
            ],
            columns=["Field", "Meaning"],
        )
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.header("Global Filters")
        ordered_min = df["test_ordered_dt"].min() if "test_ordered_dt" in df.columns else pd.NaT
        ordered_max = df["test_ordered_dt"].max() if "test_ordered_dt" in df.columns else pd.NaT
        if pd.isna(ordered_min) or pd.isna(ordered_max):
            st.error("`test_ordered_dt` is missing or invalid in the dataset.")
            st.stop()
        ordered_min_date = ordered_min.date()
        ordered_max_date = ordered_max.date()
        date_range = st.date_input(
            "Ordered Date Range",
            value=(ordered_min_date, ordered_max_date),
        )

        # Dynamic filter options: each widget only shows values present after prior selections.
        option_df = df.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            requested_start = date_range[0]
            requested_end = date_range[1]
            clamped_start = max(requested_start, ordered_min_date)
            clamped_end = min(requested_end, ordered_max_date)
            start_date = pd.to_datetime(clamped_start, utc=True)
            end_date = pd.to_datetime(clamped_end, utc=True)
            option_df = option_df[
                (option_df["test_ordered_dt"] >= start_date)
                & (option_df["test_ordered_dt"] <= end_date + pd.Timedelta(days=1))
            ]

        selected_codes = _render_dynamic_multiselect(
            "Test Code", "flt_test_code", option_df, "test_code", top_n=50
        )
        if selected_codes:
            option_df = option_df[option_df["test_code"].astype(str).isin(selected_codes)]

        selected_streets = _render_dynamic_multiselect(
            "Collection/Transit Street", "flt_event_street", option_df, "event_street", top_n=30
        )
        if selected_streets:
            option_df = option_df[option_df["event_street"].astype(str).isin(selected_streets)]

        selected_depts = _render_dynamic_multiselect(
            "Performing Lab Department", "flt_performing_dept", option_df, "test_performing_dept", top_n=30
        )
        if selected_depts:
            option_df = option_df[option_df["test_performing_dept"].astype(str).isin(selected_depts)]

        selected_locs = _render_dynamic_multiselect(
            "Performing Lab Location", "flt_performing_loc", option_df, "test_performing_location", top_n=30
        )
        if selected_locs:
            option_df = option_df[option_df["test_performing_location"].astype(str).isin(selected_locs)]

        st.markdown("---")
        st.header("A/B + Likelihood Setup")

        group_dim_options = [GROUPABLE_LABELS[c] for c in GROUPABLE_COLS]
        group_dim_label = st.selectbox("Group Dimension", options=group_dim_options, index=0)
        group_col = [k for k, v in GROUPABLE_LABELS.items() if v == group_dim_label][0]

        grp_values = _group_values_for_col(option_df, group_col)
        if not grp_values:
            st.warning("No valid groups available for the selected Group Dimension.")
            st.stop()
        if group_col == "order_day_type" and "Weekday" in grp_values and "Weekend" in grp_values:
            default_a, default_b = "Weekday", "Weekend"
        else:
            default_a = "Hospital" if "Hospital" in grp_values else grp_values[0]
            default_b = "Medical" if "Medical" in grp_values and "Medical" != default_a else (
                grp_values[1] if len(grp_values) > 1 else default_a
            )
        if "ab_group_a" in st.session_state and st.session_state["ab_group_a"] not in grp_values:
            st.session_state["ab_group_a"] = default_a
        if "ab_group_b" in st.session_state and st.session_state["ab_group_b"] not in grp_values:
            st.session_state["ab_group_b"] = default_b
        group_a = st.selectbox("Group A", options=grp_values, index=grp_values.index(default_a), key="ab_group_a")
        group_b = st.selectbox("Group B", options=grp_values, index=grp_values.index(default_b), key="ab_group_b")

        st.caption("Event likelihood is computed for: Cancellation Event")

    f = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        requested_start = date_range[0]
        requested_end = date_range[1]
        clamped_start = max(requested_start, ordered_min_date)
        clamped_end = min(requested_end, ordered_max_date)
        if requested_start != clamped_start or requested_end != clamped_end:
            st.info(
                "Selected date range exceeded available data and was auto-clamped to "
                f"{ordered_min_date} - {ordered_max_date}."
            )
        start_date, end_date = pd.to_datetime(clamped_start, utc=True), pd.to_datetime(clamped_end, utc=True)
        f = f[(f["test_ordered_dt"] >= start_date) & (f["test_ordered_dt"] <= end_date + pd.Timedelta(days=1))]
    if selected_codes:
        f = f[f["test_code"].isin(selected_codes)]
    if selected_streets:
        f = f[f["event_street"].isin(selected_streets)]
    if selected_depts:
        f = f[f["test_performing_dept"].isin(selected_depts)]
    if selected_locs:
        f = f[f["test_performing_location"].isin(selected_locs)]

    f, removed_negative_n = remove_negative_offset_records(f)
    if removed_negative_n > 0:
        st.info(f"Excluded {removed_negative_n:,} records with invalid time records.")

    if f.empty:
        st.warning("No data after filters. Please relax filter conditions.")
        st.stop()

    # Re-validate group selections against filtered cohort to avoid edge-case runtime errors.
    filtered_group_values = _group_values_for_col(f, group_col)
    if not filtered_group_values:
        st.warning("No valid groups for A/B comparison after current filters. Please relax filters.")
        st.stop()
    if group_a not in filtered_group_values:
        group_a = filtered_group_values[0]
        st.info(f"Group A was auto-adjusted to `{group_a}` for the filtered cohort.")
    if group_b not in filtered_group_values:
        group_b = filtered_group_values[1] if len(filtered_group_values) > 1 else filtered_group_values[0]
        st.info(f"Group B was auto-adjusted to `{group_b}` for the filtered cohort.")
    if group_a == group_b and len(filtered_group_values) > 1:
        group_b = next(x for x in filtered_group_values if x != group_a)
        st.info(f"Group B was auto-adjusted to `{group_b}` to keep A/B groups distinct.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unique Orders", f"{f['accession_id'].nunique():,}")
    k2.metric("Samples", f"{len(f):,}")
    total_process_median = f["offset_test_max_verified_dt_h"].dropna()
    k3.metric(
        "Median Process Time",
        f"{total_process_median.median():.2f} h" if len(total_process_median) else "N/A",
    )
    cancel_rate = f["cancellation_dt"].notna().mean()
    k4.metric("Cancellation Rate", f"{cancel_rate:.2%}")

    st.markdown("---")

    try:
        timeline_tbl = summarize_offsets(f)
        timeline_tbl = timeline_tbl[timeline_tbl["milestone"] != "cancellation_dt"].copy()
        fig1 = make_timeline_fig(timeline_tbl, "Average Timeline (Filtered Cohort)")
        st.plotly_chart(fig1, use_container_width=True)
    except Exception:
        st.warning("Unable to render average timeline for the current selection.")

    # Pre-A/B view: top-N group attainment heatmap with local threshold control.
    threshold_step, threshold_max, threshold_default = _compute_dynamic_threshold_slider_settings(f)
    # Use minute slider only for short horizons; otherwise keep hours for readability.
    if threshold_max <= 2.0:
        step_min = max(1, int(round(threshold_step * 60)))
        max_min = max(step_min, int(round(threshold_max * 60)))
        default_min = min(max_min, max(step_min, int(round(threshold_default * 60))))
        chosen_min = st.slider(
            "Attainment Threshold for Heatmap (minutes)",
            min_value=step_min,
            max_value=max_min,
            value=default_min,
            step=step_min,
        )
        attainment_threshold = chosen_min / 60.0
    else:
        attainment_threshold = st.slider(
            "Attainment Threshold for Heatmap (hours)",
            min_value=float(threshold_step),
            max_value=float(threshold_max),
            value=float(threshold_default),
            step=float(threshold_step),
            format="%.2f",
        )
    threshold_label = _format_duration_label(attainment_threshold)
    try:
        fig_attain, attainment_tbl = make_attainment_6h_heatmap(
            f, group_col=group_col, threshold_h=attainment_threshold, top_n=10
        )
        if fig_attain.data:
            st.plotly_chart(fig_attain, use_container_width=True)
            st.caption(
                "Top groups by sample count under current filters. "
                f"Cells show percent of samples reaching each stage within {threshold_label} from order placed."
            )
    except Exception:
        st.warning("Unable to render attainment heatmap for the current selection.")

    c1, c2 = st.columns(2)

    with c1:
        if group_a == group_b:
            st.info("Choose different Group A and Group B for A/B comparison.")
        else:
            try:
                fig2 = make_ab_fig(f, group_col, group_a, group_b)
                st.plotly_chart(fig2, use_container_width=True)
                ab_stats = compute_ab_completion_stats(f, group_col, group_a, group_b)
                if ab_stats.get("ok"):
                    st.caption(
                        f"A/B completion timing test (Mann-Whitney U): p-value = {ab_stats['pvalue']:.4g}. "
                        f"{ab_stats['sentence']}"
                    )
                else:
                    st.caption(f"A/B completion timing test: {ab_stats.get('reason')}")
            except Exception:
                st.warning("Unable to render A/B timing comparison for the current selection.")

    with c2:
        if group_a == group_b:
            st.info("Choose different Group A and Group B for likelihood view.")
        else:
            try:
                fig3, ldf = make_likelihood_fig(f, group_col, group_a, group_b)
                st.plotly_chart(fig3, use_container_width=True)
                like_stats = compute_likelihood_pvalue(ldf, group_a, group_b) if not ldf.empty else {"ok": False}
                if like_stats.get("ok"):
                    st.caption(
                        f"Event Likelihood Test (Two-proportion z-test): p-value = {like_stats['pvalue']:.4g}. "
                        f"{like_stats['sentence']}"
                    )
                else:
                    st.caption(f"Event Likelihood Test: {like_stats.get('reason', 'N/A')}")
                if not ldf.empty:
                    ldf_show = ldf.assign(
                        likelihood=ldf["likelihood"].map(lambda x: f"{x:.2%}"),
                        se=ldf["se"].map(lambda x: f"{x:.4f}"),
                        ci_low=ldf["ci_low"].map(lambda x: f"{x:.4f}"),
                        ci_high=ldf["ci_high"].map(lambda x: f"{x:.4f}"),
                    ).rename(
                        columns={
                            "group": "Group",
                            "defect_n": "Cancel Count",
                            "total_n": "Total Tests",
                            "likelihood": "Likelihood",
                            "se": "SE",
                            "ci_low": "CI Low",
                            "ci_high": "CI High",
                        }
                    )[
                        [
                            "Group",
                            "Cancel Count",
                            "Total Tests",
                            "Likelihood",
                            "SE",
                            "CI Low",
                            "CI High",
                        ]
                    ]
                    st.dataframe(
                        ldf_show,
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception:
                st.warning("Unable to render likelihood view for the current selection.")

    st.markdown("---")
    st.markdown("#### Filtered Cohort Snapshot (Readable Columns)")
    st.dataframe(
        f[
            [
                "accession_id",
                "test_code",
                "event_street",
                "test_performing_dept",
                "test_performing_location",
                "test_ordered_dt",
                "cancellation_dt",
            ]
        ]
        .head(200)
        .sort_values("test_ordered_dt", ascending=False),
        column_config={
            "accession_id": "Order ID",
            "test_code": "Test Code",
            "event_street": "Street",
            "test_performing_dept": "Performing Dept",
            "test_performing_location": "Performing Location",
            "test_ordered_dt": "Order Time (UTC)",
            "cancellation_dt": "Cancellation Time (UTC)",
        },
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
