import pandas as pd
import os

def load_operations_data(file_path: str = None) -> pd.DataFrame:
    # Load daily operations data from CSV
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, "logic_leap_horizon_datasets", "operations_daily_365d.csv")

        print(f"Looking for file at: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_site_meta(file_path: str = None) -> pd.DataFrame:
    # Load site metadata CSV
    if file_path is None:
        current_dir = os.getcwd()
        if os.path.basename(current_dir) in ['notebooks', 'pipeline_walkthrough']:
            project_root = os.path.dirname(current_dir)
        else:
            project_root = current_dir
        file_path = os.path.join(project_root, "logic_leap_horizon_datasets", "site_meta.csv")

        print(f"Looking for file at: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

    return pd.read_csv(file_path)

def merge_data(ops_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    # Join operations data with site info
    return ops_df.merge(meta_df, on='site_id', how='left')