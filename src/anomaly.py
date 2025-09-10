import pandas as pd
import numpy as np
from scipy.stats import zscore

def detect_anomalies_zscore(df: pd.DataFrame, column: str = 'downtime_minutes', threshold: float = 3.0) -> pd.DataFrame:
    # Z-score anomaly detection per site
    df = df.copy()
    df['zscore'] = df.groupby('site_id')[column].transform(lambda x: zscore(x, nan_policy='omit'))
    df['anomaly'] = np.abs(df['zscore']) > threshold
    return df

def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    # Create alerts from anomalies
    alerts = df[df['anomaly']][['date', 'site_id', 'downtime_minutes', 'zscore']].copy()
    alerts['alert_type'] = 'High Downtime'
    alerts['description'] = alerts.apply(
        lambda row: f"Downtime {int(row['downtime_minutes'])}min exceeds normal by {row['zscore']:.1f}sigma",
        axis=1
    )
    return alerts