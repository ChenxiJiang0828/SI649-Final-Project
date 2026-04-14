from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, norm
import streamlit as st


st.set_page_config(page_title="Specimen Journey Dashboard", layout="wide")

DATA_PATH = Path("outputs/eda/ordered_test_level_table.csv")

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
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt_cols = [c for c in df.columns if c.endswith("_dt")]
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    # Derived A/B dimension: weekday vs weekend by order timestamp.
    wd = df["test_ordered_dt"].dt.weekday
    df["order_day_type"] = wd.map(lambda x: "Weekend" if pd.notna(x) and int(x) >= 5 else "Weekday")
    return df


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
    fig.update_layout(
        title=title,
        xaxis_title="Hours Since Order Placed",
        yaxis_title="Milestone",
        height=430,
    )
    return fig


def make_ab_fig(df: pd.DataFrame, group_col: str, group_a: str, group_b: str) -> go.Figure:
    gdf = df[df[group_col].isin([group_a, group_b])].copy()
    rows = []
    for g, sub in gdf.groupby(group_col):
        s = summarize_offsets(sub)
        s["group"] = g
        rows.append(s)
    if not rows:
        return go.Figure()

    ab = pd.concat(rows, ignore_index=True)
    # Remove the baseline milestone in A/B view because it is always ~0 by definition.
    ab = ab[ab["milestone"] != "test_ordered_dt"].copy()
    ab["milestone_label"] = ab["milestone"].map(MILESTONE_LABELS).fillna(ab["milestone"])

    fig = px.bar(
        ab,
        x="median_h",
        y="milestone_label",
        color="group",
        barmode="group",
        orientation="h",
        hover_data=["n", "p25_h", "p75_h", "mean_h"],
        title=f"A/B Comparison: {GROUPABLE_LABELS.get(group_col, group_col)} ({group_a} vs {group_b})",
    )
    fig.update_layout(
        height=430,
        xaxis_title="Median Hours Since Order Placed",
        yaxis_title="Milestone",
        legend_title="Group",
    )
    return fig


def compute_ab_turnaround_stats(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    metric_col: str = "offset_test_min_verified_dt_h",
) -> dict:
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    a = sub.loc[sub[group_col] == group_a, metric_col].dropna().astype(float)
    b = sub.loc[sub[group_col] == group_b, metric_col].dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return {"ok": False, "reason": "Not enough data points for statistical testing."}

    test = mannwhitneyu(a, b, alternative="two-sided")
    med_a, med_b = float(np.median(a)), float(np.median(b))
    diff = med_a - med_b
    slower = group_a if diff > 0 else group_b
    faster = group_b if diff > 0 else group_a

    if test.pvalue < 0.05:
        sentence = (
            f"{slower} is significantly slower than {faster} "
            f"(median difference {abs(diff):.2f}h, p={test.pvalue:.3g})."
        )
    else:
        sentence = (
            f"No statistically significant turnaround difference between {group_a} and {group_b} "
            f"(median difference {abs(diff):.2f}h, p={test.pvalue:.3g})."
        )

    return {
        "ok": True,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": med_a,
        "median_b": med_b,
        "pvalue": float(test.pvalue),
        "sentence": sentence,
    }


def defect_rate(df: pd.DataFrame, mode: str, threshold_h: float = 24.0) -> pd.Series:
    if mode == "cancellation_dt":
        return df["cancellation_dt"].notna()
    return df["offset_test_min_verified_dt_h"].gt(threshold_h)


def make_likelihood_fig(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    defect_mode: str,
    threshold_h: float,
) -> tuple[go.Figure, pd.DataFrame]:
    gdf = df[df[group_col].isin([group_a, group_b])].copy()
    if gdf.empty:
        return go.Figure(), pd.DataFrame()

    gdf = gdf.assign(defect_flag=defect_rate(gdf, defect_mode, threshold_h))

    out = (
        gdf.groupby(group_col, dropna=False)["defect_flag"]
        .agg(defect_n="sum", total_n="count")
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    out["likelihood"] = out["defect_n"] / out["total_n"]
    # 95% CI (normal approximation) for a binomial proportion.
    out["se"] = (out["likelihood"] * (1 - out["likelihood"]) / out["total_n"]).pow(0.5)
    out["ci_low"] = (out["likelihood"] - 1.96 * out["se"]).clip(lower=0)
    out["ci_high"] = (out["likelihood"] + 1.96 * out["se"]).clip(upper=1)

    title_event = "cancellation_dt" if defect_mode == "cancellation_dt" else f"slow turnaround > {threshold_h:.0f}h"

    # Dot plot with confidence intervals is more readable for 2-group A/B.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=out["group"],
            y=out["likelihood"],
            mode="markers+text",
            text=[f"{x:.1%}" for x in out["likelihood"]],
            textposition="top center",
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
                "Defect Count=%{customdata[0]}<br>"
                "Total=%{customdata[1]}<extra></extra>"
            ),
            name="Defect Likelihood",
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

    fig.update_layout(
        title=f"Event Likelihood ({title_event})",
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


def main() -> None:
    st.title("Specimen Journey Dashboard (V1)")
    st.caption("Interactive dashboard for average timeline, A/B comparison, and defect-event likelihood.")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    df = load_data(str(DATA_PATH))

    with st.expander("Data Dictionary (What Each Variable Means)", expanded=False):
        dict_df = pd.DataFrame(
            [
                ["accession_id", "Lab order ID (one order can include multiple test codes)."],
                ["test_code", "Ordered test code; analysis grain uses accession_id + test_code."],
                ["event_street", "Street-level collection/transit location tied to the event."],
                ["test_performing_dept", "Lab department responsible for running the test."],
                ["test_performing_location", "Physical location of the performing lab."],
                ["test_ordered_dt", "Timestamp when the order was placed."],
                ["offset_test_min_verified_dt_h", "Turnaround hours from order to first verified result."],
                ["cancellation_dt", "Timestamp when the ordered test was cancelled (if applicable)."],
                ["Slow Turnaround", "Derived defect: turnaround hours > selected threshold."],
            ],
            columns=["Field", "Meaning"],
        )
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.header("Global Filters")
        ordered_min = df["test_ordered_dt"].min()
        ordered_max = df["test_ordered_dt"].max()
        date_range = st.date_input(
            "Ordered Date Range",
            value=(ordered_min.date(), ordered_max.date()),
            min_value=ordered_min.date(),
            max_value=ordered_max.date(),
        )

        top_test_codes = df["test_code"].value_counts().head(50).index.tolist()
        selected_codes = st.multiselect("Test Code (Top 50 by volume)", options=top_test_codes, default=[])

        top_streets = df["event_street"].value_counts().head(30).index.tolist()
        selected_streets = st.multiselect("Collection/Transit Street (Top 30)", options=top_streets, default=[])

        top_depts = df["test_performing_dept"].value_counts().head(30).index.tolist()
        selected_depts = st.multiselect("Performing Lab Department (Top 30)", options=top_depts, default=[])

        top_locs = df["test_performing_location"].value_counts().head(30).index.tolist()
        selected_locs = st.multiselect("Performing Lab Location (Top 30)", options=top_locs, default=[])

        st.markdown("---")
        st.header("A/B + Likelihood Setup")

        group_dim_options = [GROUPABLE_LABELS[c] for c in GROUPABLE_COLS]
        group_dim_label = st.selectbox("Group Dimension", options=group_dim_options, index=0)
        group_col = [k for k, v in GROUPABLE_LABELS.items() if v == group_dim_label][0]

        grp_values = sorted([x for x in df[group_col].dropna().astype(str).unique().tolist()])
        if group_col == "order_day_type" and "Weekday" in grp_values and "Weekend" in grp_values:
            default_a, default_b = "Weekday", "Weekend"
        else:
            default_a = "Hospital" if "Hospital" in grp_values else grp_values[0]
            default_b = "Medical" if "Medical" in grp_values and "Medical" != default_a else (
                grp_values[1] if len(grp_values) > 1 else default_a
            )
        group_a = st.selectbox("Group A", options=grp_values, index=grp_values.index(default_a))
        group_b = st.selectbox("Group B", options=grp_values, index=grp_values.index(default_b))

        defect_mode_label = st.selectbox(
            "Defect Event",
            options=["Cancellation Event", "Slow Turnaround"],
            help="Slow Turnaround means: order -> first verified result is greater than threshold.",
        )
        slow_threshold = st.slider("Slow Turnaround Threshold (hours)", 6, 96, 24, 2)

    f = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0], utc=True), pd.to_datetime(date_range[1], utc=True)
        f = f[(f["test_ordered_dt"] >= start_date) & (f["test_ordered_dt"] <= end_date + pd.Timedelta(days=1))]
    if selected_codes:
        f = f[f["test_code"].isin(selected_codes)]
    if selected_streets:
        f = f[f["event_street"].isin(selected_streets)]
    if selected_depts:
        f = f[f["test_performing_dept"].isin(selected_depts)]
    if selected_locs:
        f = f[f["test_performing_location"].isin(selected_locs)]

    if f.empty:
        st.warning("No data after filters. Please relax filter conditions.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ordered Tests (Filtered)", f"{len(f):,}")
    k2.metric("Unique Orders", f"{f['accession_id'].nunique():,}")
    med_tat = f["offset_test_min_verified_dt_h"].dropna().median()
    k3.metric("Median Turnaround (Order -> First Verified)", f"{med_tat:.2f} h" if pd.notna(med_tat) else "N/A")
    cancel_rate = f["cancellation_dt"].notna().mean()
    k4.metric("Cancellation Rate", f"{cancel_rate:.2%}")

    st.markdown("---")

    t = summarize_offsets(f)
    fig1 = make_timeline_fig(t, "1) Average Timeline (Filtered Cohort)")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        if group_a == group_b:
            st.info("Choose different Group A and Group B for A/B comparison.")
        else:
            fig2 = make_ab_fig(f, group_col, group_a, group_b)
            st.plotly_chart(fig2, use_container_width=True)
            ab_stats = compute_ab_turnaround_stats(f, group_col, group_a, group_b)
            if ab_stats.get("ok"):
                st.caption(
                    f"A/B Turnaround Test (Mann-Whitney U): p-value = {ab_stats['pvalue']:.4g}. "
                    f"{ab_stats['sentence']}"
                )
            else:
                st.caption(f"A/B Turnaround Test: {ab_stats.get('reason')}")

    with c2:
        if group_a == group_b:
            st.info("Choose different Group A and Group B for likelihood view.")
        else:
            defect_mode = "cancellation_dt" if defect_mode_label == "Cancellation Event" else "slow_turnaround"
            fig3, ldf = make_likelihood_fig(f, group_col, group_a, group_b, defect_mode, slow_threshold)
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
                st.dataframe(
                    ldf.assign(likelihood=ldf["likelihood"].map(lambda x: f"{x:.2%}")).rename(
                        columns={
                            "group": "Group",
                            "defect_n": "Defect Count",
                            "total_n": "Total Ordered Tests",
                            "likelihood": "Defect Likelihood",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    st.subheader("Filtered Cohort Snapshot (Readable Columns)")
    st.dataframe(
        f[
            [
                "accession_id",
                "test_code",
                "event_street",
                "test_performing_dept",
                "test_performing_location",
                "test_ordered_dt",
                "offset_test_min_verified_dt_h",
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
            "offset_test_min_verified_dt_h": "Turnaround Hours (Order -> First Verified)",
            "cancellation_dt": "Cancellation Time (UTC)",
        },
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
