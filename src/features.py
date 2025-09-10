import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Flag holidays (zero production days)
    df = df.copy()
    df['is_holiday'] = (df['units_produced'] == 0) & (df['holiday_flag'] == 1)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time features
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Lag features (1,7,14,30 days)
    for lag in [1, 7, 14, 30]:
        df[f'units_produced_lag_{lag}'] = df.groupby('site_id')['units_produced'].shift(lag)
        df[f'power_kwh_lag_{lag}'] = df.groupby('site_id')['power_kwh'].shift(lag)

    # Rolling averages (7,14,30 day windows)
    for window in [7, 14, 30]:
        df[f'units_produced_roll_mean_{window}'] = df.groupby('site_id')['units_produced'].rolling(window).mean().reset_index(0, drop=True)
        df[f'power_kwh_roll_mean_{window}'] = df.groupby('site_id')['power_kwh'].rolling(window).mean().reset_index(0, drop=True)

    # Derived features
    df['efficiency'] = df['units_produced'] / (df['power_kwh'] + 1e-6)  # units per kWh
    df['downtime_ratio'] = df['downtime_minutes'] / (df['shift_hours_per_day'] * 60)

    # Backfill NaNs from early lags
    df.fillna(method='bfill', inplace=True)
    return df