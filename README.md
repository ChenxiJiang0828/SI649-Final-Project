# SI 649 Final Project: Specimen Journey Dashboard

This project analyzes lab specimen journey events and provides:
- Reproducible EDA outputs
- A draft PDF report section (`Introduction` + `EDA`)
- An interactive dashboard with:
  - Average timeline view
  - A/B comparison view
  - Event likelihood view (with p-values and significance summary)

## Project Structure

- `2025_specimen_time_series_events_no_phi.tsv`: source dataset
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
- Defect event options:
  - Cancellation event
  - Slow turnaround (threshold configurable)
- Statistical outputs:
  - A/B turnaround: Mann-Whitney U test + p-value + natural-language significance summary
  - Event likelihood: two-proportion z-test + p-value + natural-language significance summary

## Notes

- Main analysis grain is **ordered test**: `accession_id + test_code`.
- `Order Placed` is excluded in A/B chart because it is the zero-time baseline by definition.
- If filters produce no rows, relax filters in the sidebar.

## Reproducibility Checklist

1. `pip install -r requirements.txt`
2. `python scripts/eda_specimen.py`
3. `python scripts/generate_intro_eda_pdf.py` (optional for report draft)
4. `streamlit run dashboard_app.py`

