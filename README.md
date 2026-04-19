# SI649 Final Project: Specimen Journey Dashboard

## Links
- GitHub repository: https://github.com/ChenxiJiang0828/SI649-Final-Project
- Deployed app: https://si649-final-project-c9i8szukdpkppytkexbb2d.streamlit.app/

## End-to-End Data + App Workflow

This project supports a true end-to-end flow starting from the original raw file:

- Raw source file: `2025_specimen_time_series_events_no_phi.tsv`
- Dashboard app: `dashboard_app.py`
- Data split helper for GitHub: `scripts/split_raw_tsv_for_github.py`

The dashboard can run from either:
1. prebuilt ordered-test table (if present), or
2. raw single TSV file, or
3. split raw TSV parts in `data/raw_parts/`.

## Setup

Use Python 3.10+.

```bash
cd <your_project_path>
pip install -r requirements.txt
```

## How to Run the Dashboard

1. Make sure at least one data source exists:
   - `outputs/eda/ordered_test_level_table.csv` (or `.parquet`), OR
   - `2025_specimen_time_series_events_no_phi.tsv`, OR
   - `data/raw_parts/2025_specimen_time_series_events_no_phi.part*.tsv`

2. Start the app:

```bash
streamlit run dashboard_app.py
```

3. Open the local URL shown in terminal (usually `http://localhost:8501`).

If the browser does not open automatically, copy the URL from terminal and open it manually.

## Split Raw TSV for GitHub Push

GitHub rejects files larger than 100MB.  
Use the script below to split the raw TSV into parts:

```bash
python scripts/split_raw_tsv_for_github.py --input 2025_specimen_time_series_events_no_phi.tsv --output-dir data/raw_parts --max-mb 95
```

This creates files like:
- `data/raw_parts/2025_specimen_time_series_events_no_phi.part1.tsv`
- `data/raw_parts/2025_specimen_time_series_events_no_phi.part2.tsv`
- `data/raw_parts/2025_specimen_time_series_events_no_phi.part3.tsv`

## Included Code

- `dashboard_app.py`: Streamlit dashboard (overall timeline, stage-attainment heatmap, A/B comparison, cancellation likelihood)
- `scripts/split_raw_tsv_for_github.py`: split raw TSV into GitHub-friendly part files
- `scripts/eda_specimen.py`: EDA pipeline script
- `scripts/timestamp_quality_checks.py`: timestamp quality check script
- `scripts/generate_intro_eda_pdf.py`: report PDF generation helper
