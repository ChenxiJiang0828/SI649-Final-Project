# SI649 Final Project: Specimen Journey Dashboard

Code repository: https://github.com/ChenxiJiang0828/SI649-Final-Project.git  
Deployed app: https://si649-final-project-c9i8szukdpkppytkexbb2d.streamlit.app/

## Setup

Use Python 3.10+.

```bash
cd <your_project_path>
pip install -r requirements.txt
```

## Run

1. Build the ordered-test table:

```bash
python scripts/build_ordered_test_table.py
```

This script performs preprocessing and generates the wide, precomputed table used by the dashboard:
- `outputs/eda/ordered_test_level_table.parquet`

2. Start the dashboard:

```bash
streamlit run dashboard_app.py
```

3. Open the URL shown in terminal (usually `http://localhost:8501`).

## Code for EDA

`scripts/timestamp_quality_checks.py` runs timestamp quality checks used in analysis/discussion.

Input:
- `outputs/eda/ordered_test_level_table.parquet` (or `.csv`) if available
- The script filters to ordered tests in year 2025 based on `test_ordered_dt`

Run:

```bash
python scripts/timestamp_quality_checks.py
```

Outputs:
- `outputs/eda/timestamp_cross_stage_violations.csv`
  - Cross-stage order violations (e.g., later stage timestamp earlier than prior stage)
- `outputs/eda/timestamp_glucm_equalities.csv`
  - GLUCM-specific timestamp equality checks across milestones
- `outputs/eda/timestamp_quality_summary.txt`
  - Human-readable summary of key counts and percentages
