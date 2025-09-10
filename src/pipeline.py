import pandas as pd
from loader import load_operations_data, load_site_meta, merge_data
from features import clean_data, engineer_features
from models import train_models, forecast_future
from anomaly import detect_anomalies_zscore, generate_alerts
import numpy as np

def run_pipeline():
    # Load and merge data
    ops_df = load_operations_data()
    meta_df = load_site_meta()
    df = merge_data(ops_df, meta_df)
    
    # Process data
    df = clean_data(df)
    df = engineer_features(df)
    
    # Model features list
    features = ['day_of_week', 'month', 'quarter', 'day_of_year', 'week_of_year',
                'units_produced_lag_1', 'units_produced_lag_7', 'units_produced_lag_14', 'units_produced_lag_30',
                'units_produced_roll_mean_7', 'units_produced_roll_mean_14', 'units_produced_roll_mean_30',
                'power_kwh_lag_1', 'power_kwh_lag_7', 'power_kwh_lag_14', 'power_kwh_lag_30',
                'power_kwh_roll_mean_7', 'power_kwh_roll_mean_14', 'power_kwh_roll_mean_30',
                'efficiency', 'downtime_ratio', 'temperature_c', 'rainfall_mm', 'holiday_flag']
    
    # Train models
    models_units = train_models(df, 'units_produced', features)
    models_power = train_models(df, 'power_kwh', features)
    
    # Generate forecasts
    forecasts_units = forecast_future(df, models_units, 'units_produced', 14)
    forecasts_power = forecast_future(df, models_power, 'power_kwh', 14)
    
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=14, freq='D')
    
    # Combined forecast output
    forecast_records = []
    for site in forecasts_units:
        for i, (units_val, power_val) in enumerate(zip(forecasts_units[site], forecasts_power[site])):
            forecast_records.append({
                'date': future_dates[i],
                'site_id': site,
                'forecast_units': units_val,
                'forecast_power': power_val
            })
    
    pd.DataFrame(forecast_records).to_csv('outputs/forecasts.csv', index=False)
    
    # Separate outputs (compatibility)
    forecast_units_df = [{'date': future_dates[i], 'site_id': site, 'forecast_units': val}
                        for site in forecasts_units for i, val in enumerate(forecasts_units[site])]
    forecast_power_df = [{'date': future_dates[i], 'site_id': site, 'forecast_power': val}
                        for site in forecasts_power for i, val in enumerate(forecasts_power[site])]
    
    pd.DataFrame(forecast_units_df).to_csv('outputs/forecast_units.csv', index=False)
    pd.DataFrame(forecast_power_df).to_csv('outputs/forecast_power.csv', index=False)
    
    # Anomaly detection
    df_anomaly = detect_anomalies_zscore(df)
    alerts_df = generate_alerts(df_anomaly)
    alerts_df.to_csv('outputs/alerts.csv', index=False)
    
    # Print model performance
    for site in models_units:
        print(f"Site {site} - Units: LR MAE {models_units[site]['avg_mae_lr']:.2f}, XGB MAE {models_units[site]['avg_mae_xgb']:.2f}")
    for site in models_power:
        print(f"Site {site} - Power: LR MAE {models_power[site]['avg_mae_lr']:.2f}, XGB MAE {models_power[site]['avg_mae_xgb']:.2f}")

if __name__ == "__main__":
    run_pipeline()