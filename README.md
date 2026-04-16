# SI 649 Final Project: Specimen Journey Dashboard

This project analyzes lab specimen journey events and provides:
- Reproducible EDA outputs
- A draft PDF report section (`Introduction` + `EDA`)
- An interactive dashboard with:
  - Stage completion curves (time vs % completed)
  - Top-group attainment heatmap (threshold-adjustable)
  - A/B comparison view
  - Event likelihood view (with p-values and significance summary)

## Project Structure

- `2025_specimen_time_series_events_no_phi.tsv`: original raw source dataset (local)
- `data/raw_parts/*.tsv`: split raw dataset parts for GitHub-compatible storage
- `requirements.txt`: Python dependencies
- `scripts/eda_specimen.py`: EDA pipeline + intermediate outputs
- `scripts/generate_intro_eda_pdf.py`: generates `report_intro_eda.pdf`
- `dashboard_app.py`: Streamlit dashboard app
- `outputs/eda/`: generated EDA files (csv/json/png)

## Setup

1. Use Python 3.10+ (tested with Python 3.13).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run EDA

Generate EDA outputs used by the dashboard and report:

```bash
python scripts/eda_specimen.py
```

Default behavior:
- Uses `2025_specimen_time_series_events_no_phi.tsv`
- Filters to year 2025
- Writes outputs to `outputs/eda/`

Optional example:

```bash
python scripts/eda_specimen.py --filter-column event_street --filter-value Hospital --ab-group-col test_performing_location --group-a ULAB --group-b NHLA
```

## Generate Report PDF (Introduction + EDA)

```bash
python scripts/generate_intro_eda_pdf.py
```

Output:
- `report_intro_eda.pdf`

## Run Interactive Dashboard

```bash
streamlit run dashboard_app.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).

### Data Loading Order (Dashboard)

`dashboard_app.py` auto-loads data in this order:
1. `outputs/eda/ordered_test_level_table.parquet`
2. `outputs/eda/ordered_test_level_table.csv`
3. `data/ordered_test_level_table.parquet`
4. `data/ordered_test_level_table.csv`
5. If none exist, build on startup from `data/raw_parts/2025_specimen_time_series_events_no_phi.part*.tsv`

## Dashboard Features

- Global filters:
  - Ordered date range
  - Test code
  - Street
  - Performing department
  - Performing location
- A/B grouping options:
  - Weekday vs Weekend (`Order Day Type`)
  - Street / department / location / test code
- Attainment heatmap control (in main area):
  - `Attainment Threshold for Heatmap (hours)` with 0.5-hour step
- Defect event (current dashboard version):
  - Cancellation event
- Statistical outputs:
  - A/B completion timing (`Order -> Final Verified`): Mann-Whitney U test + p-value + natural-language significance summary
  - Event likelihood: two-proportion z-test + p-value + natural-language significance summary
- Robust date behavior:
  - If selected date range exceeds data bounds, it is auto-clamped to available min/max dates.

## Notes

- Main analysis grain is **ordered test**: `accession_id + test_code`.
- `Order Placed` is excluded in A/B chart because it is the zero-time baseline by definition.
- Completion curves are displayed for the first 15 hours after order placement.
- If filters produce no rows, relax filters in the sidebar.
- Raw dataset is split into multiple TSV parts because GitHub rejects single files larger than 100MB.

## Reproducibility Checklist

1. `pip install -r requirements.txt`
2. `python scripts/eda_specimen.py`
3. `python scripts/generate_intro_eda_pdf.py` (optional for report draft)
4. `streamlit run dashboard_app.py`
