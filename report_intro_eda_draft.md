# Introduction

Clinical laboratory testing follows a multi-step specimen journey: a physician places an order, a specimen is collected, transported through the lab network, received by the performing lab, analyzed, and finally reported/verified back to care teams. This journey is inherently temporal and operational, and delays or defects at any step can directly affect treatment timing, throughput, and resource utilization.

This project focuses on making that journey inspectable at scale using event-level time-series data. In addition to normal process events (for example `test_ordered_dt`, `test_collected_dt`, `test_receipt_dt`, and result verification events), the data also captures potential defects such as order cancellation (`cancellation_dt`) and many tube-tracker movement events. The goal is not only to summarize average process behavior, but also to support targeted comparisons and defect risk assessment under different operational conditions.

The primary stakeholders are:
- Clinical operations and laboratory managers, who need to monitor turnaround time and process stability.
- Quality/safety teams, who need to detect where defects are more likely and prioritize interventions.
- Frontline clinical leaders, who need quick comparisons across cohorts (for example location- or test-specific groups) to guide workflow and staffing decisions.

The visualization supports three concrete decision tasks:
- Average timeline inspection under dimensional filters: “What does a typical ordered-test timeline look like for a specific subgroup?”
- A/B comparison: “Which of two mutually exclusive cohorts shows slower or more variable progression?”
- Event likelihood analysis: “How likely is a named defect event in each cohort?”

Our approach was to (1) build a clean ordered-test-level analysis table keyed by `accession_id + test_code`, (2) compute milestone offsets relative to `test_ordered_dt`, (3) profile data completeness and distribution shape before design, and (4) use robust summaries (median/IQR, percentile-based comparisons) for timeline and cohort views due long-tailed timing behavior.

# Exploratory Data Analysis

## Dataset Summary

Using the 2025 extract only, the dataset contains **2,174,612 rows** and **13 columns**, spanning **2025-01-01 to 2025-12-31** (`outputs/eda/basic_profile.json`).

Key cardinalities:
- `accession_id`: 139,932
- `test_code`: 1,022
- `event_type`: 56
- `event_source`: 2 (`order`, `tube_tracker`)

Because ordered-test events are defined at the level of test code on an order, the analysis grain for process timing is `accession_id + test_code` (ordered test), not accession alone.

## Event Counts and Process Coverage

Top process events by count (`outputs/eda/event_type_counts.csv`) show the expected backbone of the specimen journey:
- `test_ordered_dt`: 236,292
- `test_collected_dt`: 230,132
- `test_receipt_dt`: 233,444
- `test_min_resulted_dt`: 224,295
- `test_min_verified_dt`: 225,790
- `cancellation_dt`: 43,109 (raw event count)

At ordered-test grain (`outputs/eda/milestone_coverage_ordered_test.csv`), milestone coverage is high for core flow:
- ordered: 100.0%
- collected: 99.60%
- receipt: 97.46%
- min verified: 94.46%
- cancellation: 4.32%

This indicates most tests follow a complete order-to-verification path, with a smaller but meaningful defect subset.

## Null Values and Data Quality

Column-level missingness (`outputs/eda/column_summary.csv`) is low overall:
- `barcode`: 79,859 missing (3.67%)
- `test_code`: 2,320 missing (0.11%)
- all other columns: 0% missing in this extract

The low null rate supports robust cohort slicing, while barcode missingness suggests tube-level identifiers should be used carefully in analyses that depend strictly on barcode completeness.

## Time Distributions and Early Observations

From `outputs/eda/timeline_average_all.csv`, median milestone offsets (hours since `test_ordered_dt`) are:
- collected: 0.10 h
- receipt: 0.72 h
- min resulted: 1.25 h
- min verified: 1.40 h
- max verified: 1.58 h
- cancellation: 1.75 h (for canceled subset)

Two important distributional observations shaped design:
- Timing is long-tailed (means are much larger than medians for later milestones), so medians and IQR are more appropriate than means for the main timeline story.
- Some offsets are negative in aggregate summaries (for example mean collected offset), likely from timestamp/ordering artifacts across systems. This motivates robust summaries and explicit QC notes in interpretation.

Filtered example (`outputs/eda/timeline_average_filter_test_code_CBCD.csv`) shows how subgroup behavior differs: for `test_code=CBCD`, median verification timing is tighter and generally earlier than global aggregates, demonstrating why dimensional filtering is essential.

## How EDA Informed Visualization Design

EDA directly informed the three required views:
- **Average timeline view**: Use milestone medians with IQR error bars (not means), anchored at `test_ordered_dt`, with interactive filters on `test_code`, `event_street`, `test_performing_dept`, and `test_performing_location`.
- **A/B comparison view**: Compare matched milestone medians across two mutually exclusive groups (default example: `Hospital` vs `Medical`) using linked point/segment encoding to make stage-wise gaps immediately visible.
- **Event likelihood view**: Start with a clear defect event (`cancellation_dt`) and show normalized likelihood by group, because raw counts are confounded by cohort size.

`outputs/eda/event_likelihood_ab.csv` already shows a meaningful cohort difference:
- Hospital: 4.99% cancellation likelihood (3023 / 60610)
- Medical: 4.07% cancellation likelihood (6116 / 150159)

This confirms that defect-likelihood comparison can surface actionable differences that are not obvious from timeline averages alone.

## Annotated Screenshot Guidance (for Report Figures)

Use the generated files in `outputs/eda/` and annotate each screenshot with short callouts:

- **Figure EDA-1 (Data Profile)**: screenshot/table from `column_summary.csv` + `basic_profile.json`
  - Callout A: total rows/columns and 2025 time span
  - Callout B: barcode missingness (3.67%) as main null-value caveat

- **Figure EDA-2 (Event Composition)**: chart/table from `event_type_counts.csv`
  - Callout A: dominant ordered-test milestones
  - Callout B: presence of both `order` and `tube_tracker` processes

- **Figure EDA-3 (Global Timeline Distribution)**: `timeline_average_all.csv` (or `timeline_test_code=CBCD.png` if discussing filtered case)
  - Callout A: median path from ordered to verified
  - Callout B: long-tail behavior (mean vs median gap)

- **Figure EDA-4 (Filter/A-B/Defect Readiness)**: `timeline_average_filter_test_code_CBCD.csv` + `event_likelihood_ab.csv`
  - Callout A: filtered timeline differs from global timeline
  - Callout B: cancellation likelihood differs across Hospital vs Medical
