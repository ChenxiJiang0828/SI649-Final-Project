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
