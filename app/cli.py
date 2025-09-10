import typer
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import os
import sys

# Add project root to path
project_root = os.getcwd()
if os.path.basename(project_root) in ['app', 'notebooks']:
    project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from src.loader import load_operations_data, load_site_meta, merge_data
from src.features import clean_data, engineer_features
from src.models import train_models, forecast_future
from src.anomaly import detect_anomalies_zscore, generate_alerts

app = typer.Typer(help="Forecast + Alerting Pipeline CLI for multi-site operations")

def load_pipeline_data(target_date: Optional[str] = None, force_retrain: bool = False):
    # Load data and train models
    print("Loading pipeline data...")
    
    ops_df = load_operations_data()
    meta_df = load_site_meta()
    df = merge_data(ops_df, meta_df)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    
    feature_cols = [
        'day_of_week', 'month', 'quarter', 'day_of_year', 'week_of_year',
        'units_produced_lag_1', 'units_produced_lag_7', 'units_produced_lag_14', 'units_produced_lag_30',
        'units_produced_roll_mean_7', 'units_produced_roll_mean_14', 'units_produced_roll_mean_30',
        'power_kwh_lag_1', 'power_kwh_lag_7', 'power_kwh_lag_14', 'power_kwh_lag_30',
        'power_kwh_roll_mean_7', 'power_kwh_roll_mean_14', 'power_kwh_roll_mean_30',
        'efficiency', 'downtime_ratio', 'temperature_c', 'rainfall_mm', 'is_holiday'
    ]
    
    print("Training models...")
    models_units = train_models(df_features, 'units_produced', feature_cols, force_retrain=force_retrain)
    models_power = train_models(df_features, 'power_kwh', feature_cols, force_retrain=force_retrain)
    
    print("Detecting anomalies...")
    df_anomalies = detect_anomalies_zscore(df_features, 'downtime_minutes', 3.0)
    alerts_df = generate_alerts(df_anomalies)
    
    print("Pipeline data loaded successfully!")
    return df_features, models_units, models_power, alerts_df, feature_cols


def generate_forecast_data(site: str, days: int, df_features, models_units, models_power):
    # Create forecast DataFrame for site
    if site not in models_units:
        typer.echo(f"Site {site} not found in trained models. Available sites: {list(models_units.keys())}")
        raise typer.Exit(code=1)
    
    forecasts_units = forecast_future(df_features, {site: models_units[site]}, 'units_produced', days)
    forecasts_power = forecast_future(df_features, {site: models_power[site]}, 'power_kwh', days)
    
    last_date = df_features['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
    
    forecast_data = []
    for i, (units_val, power_val) in enumerate(zip(forecasts_units[site], forecasts_power[site])):
        forecast_data.append({
            'date': future_dates[i].strftime('%Y-%m-%d'),
            'site_id': site,
            'forecast_units': round(units_val, 2),
            'forecast_power': round(power_val, 2)
        })
    
    return pd.DataFrame(forecast_data)


def process_anomalies(site: str, start_date: Optional[str], end_date: Optional[str],
                      severity: Optional[str], alerts_df):
    # Filter and process anomalies for site
    if site not in alerts_df['site_id'].values:
        typer.echo(f"No anomaly data found for site {site}")
        raise typer.Exit(code=1)
    
    site_alerts = alerts_df[alerts_df['site_id'] == site].copy()
    
    if start_date or end_date:
        site_alerts['date'] = pd.to_datetime(site_alerts['date'])
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            site_alerts = site_alerts[site_alerts['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            site_alerts = site_alerts[site_alerts['date'] <= end_dt]
        
        if site_alerts.empty:
            typer.echo(f"No anomalies found for site {site} in date range {start_date} to {end_date}")
            return pd.DataFrame()
    
    if 'severity' not in site_alerts.columns:
        site_alerts['severity'] = site_alerts['zscore'].apply(
            lambda z: 'high' if z > 3.0 else 'medium' if z > 2.0 else 'low'
        )
    
    if severity and not site_alerts.empty:
        site_alerts = site_alerts[site_alerts['severity'].str.lower() == severity.lower()]
        
        if site_alerts.empty:
            typer.echo(f"No {severity} severity anomalies found for site {site}")
            return pd.DataFrame()
    
    return site_alerts.sort_values(['date', 'zscore'], ascending=[True, False])

def output_forecast(forecast_df: pd.DataFrame, site: str, days: int, format: str):
    # Print forecast results
    if format == "table":
        typer.echo(f"\n{days}-DAY FORECAST FOR SITE {site.upper()}")
        typer.echo("=" * 60)
        for _, row in forecast_df.iterrows():
            typer.echo(f"{row['date']:<12} | Units: {row['forecast_units']:>8.1f} | Power: {row['forecast_power']:>8.1f} kWh")
        
        avg_units = forecast_df['forecast_units'].mean()
        avg_power = forecast_df['forecast_power'].mean()
        trend_units = forecast_df['forecast_units'].iloc[-1] - forecast_df['forecast_units'].iloc[0]
        trend_power = forecast_df['forecast_power'].iloc[-1] - forecast_df['forecast_power'].iloc[0]
        
        typer.echo("\nSUMMARY:")
        typer.echo(f"   Average Units/Day:  {avg_units:.1f}")
        typer.echo(f"   Average Power/Day:  {avg_power:.1f} kWh")
        typer.echo(f"   Units Trend ({days} days): {trend_units:+.1f}")
        typer.echo(f"   Power Trend ({days} days): {trend_power:+.1f} kWh")
        
    elif format == "json":
        typer.echo(forecast_df.to_json(orient='records', date_format='iso', indent=2))
        
    elif format == "csv":
        csv_output = forecast_df.to_csv(index=False)
        typer.echo(csv_output)
        output_file = f"forecast_{site}_{days}d.csv"
        forecast_df.to_csv(output_file, index=False)
        typer.echo(f"\nSaved to {output_file}")
    
    else:
        typer.echo(f"Unsupported format: {format}. Use 'table', 'json', or 'csv'.")
        raise typer.Exit(code=1)


@app.command()
def forecast(
    site: str = typer.Argument(..., help="Site ID (e.g., S1, S2)"),
    days: int = typer.Option(14, "--days", "-d", help="Number of days to forecast (default: 14)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="Force retraining of models (ignore cache)")
):
    # Forecast command
    df_features, models_units, models_power, _, _ = load_pipeline_data(force_retrain=force_retrain)
    
    print(f"Generating {days}-day forecast for site {site}...")
    forecast_df = generate_forecast_data(site, days, df_features, models_units, models_power)
    output_forecast(forecast_df, site, days, format)

def output_anomalies(site_alerts: pd.DataFrame, site: str, start_date: Optional[str],
                     end_date: Optional[str], format: str):
    # Print anomaly results
    if site_alerts.empty:
        if format == "table":
            typer.echo("No anomalies detected for this site/date range.")
        return
    
    typer.echo(f"\nANOMALY ALERTS FOR SITE {site.upper()}")
    if start_date and end_date:
        typer.echo(f"Date Range: {start_date} to {end_date}")
    typer.echo("=" * 80)
    
    if format == "table":
        for _, alert in site_alerts.iterrows():
            severity_symbol = {"low": "LOW", "medium": "MED", "high": "HIGH"}.get(alert['severity'], "N/A")
            date_str = alert['date'].strftime('%Y-%m-%d') if hasattr(alert['date'], 'strftime') else str(alert['date'])
            clean_desc = str(alert['description'])
            
            typer.echo(f"{date_str:<12} | Z-Score: {alert['zscore']:>5.1f} | "
                      f"Downtime: {alert['downtime_minutes']:>6.0f}min | "
                      f"Severity: {severity_symbol:<7} | "
                      f"{clean_desc[:60]}{'...' if len(clean_desc) > 60 else ''}")
        
        total_alerts = len(site_alerts)
        high_severity = len(site_alerts[site_alerts['severity'] == 'high'])
        avg_downtime = site_alerts['downtime_minutes'].mean()
        
        typer.echo(f"\nSUMMARY: {total_alerts} total alerts, {high_severity} high-severity")
        typer.echo(f"Average downtime per anomaly: {avg_downtime:.0f} minutes")
    
    elif format == "json":
        site_alerts_json = site_alerts.copy()
        if 'date' in site_alerts_json.columns:
            if hasattr(site_alerts_json['date'].iloc[0], 'strftime'):
                site_alerts_json['date'] = site_alerts_json['date'].dt.strftime('%Y-%m-%d')
            elif isinstance(site_alerts_json['date'].iloc[0], str):
                pass
            else:
                site_alerts_json['date'] = site_alerts_json['date'].astype(str)
        
        typer.echo(site_alerts_json.to_json(orient='records', indent=2))
        
    elif format == "csv":
        csv_output = site_alerts.to_csv(index=False)
        typer.echo(csv_output)
        filename = f"anomalies_{site}"
        if start_date and end_date:
            filename += f"_{start_date}_to_{end_date}"
        filename += ".csv"
        site_alerts.to_csv(filename, index=False)
        typer.echo(f"\nSaved to {filename}")
    
    else:
        typer.echo(f"Unsupported format: {format}. Use 'table', 'json', or 'csv'.")
        raise typer.Exit(code=1)


@app.command()
def anomalies(
    site: str = typer.Argument(..., help="Site ID (e.g., S1, S2)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", "-s", help="Start date (DD-MM-YYYY)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (DD-MM-YYYY)"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter by severity: low, medium, high"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="Force retraining of models (ignore cache)")
):
    # Anomaly command
    _, _, _, alerts_df, _ = load_pipeline_data(force_retrain=force_retrain)
    
    site_alerts = process_anomalies(site, start_date, end_date, severity, alerts_df)
    if site_alerts is None:
        return
    
    output_anomalies(site_alerts, site, start_date, end_date, format)

@app.command()
def overview(
    site: str = typer.Argument(..., help="Site ID (e.g., S1, S2)"),
    days: int = typer.Option(14, "--days", "-d", help="Forecast days (default: 14)"),
    recent_days: int = typer.Option(30, "--recent-days", "-r", help="Recent days for anomalies (default: 30)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="Force retraining of models (ignore cache)")
):
    # Overview: forecasts + recent anomalies
    typer.echo(f"\nCOMPREHENSIVE OVERVIEW FOR SITE {site.upper()}")
    typer.echo("=" * 60)
    
    print(f"\n{'='*20} FORECASTS {'='*20}")
    forecast(site=site, days=days, format=format, force_retrain=force_retrain)
    typer.echo("\n" + "="*60 + "\n")
    
    from datetime import datetime
    end_date = (datetime.now() - timedelta(days=recent_days)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=recent_days*2)).strftime('%Y-%m-%d')
    
    print(f"\n{'='*20} RECENT ANOMALIES {'='*20}")
    anomalies(site=site, start_date=start_date, end_date=end_date, format=format, force_retrain=force_retrain)

@app.command()
def sites():
    # List available sites
    print("LOADING SITE METADATA...")
    meta_df = load_site_meta()
    
    typer.echo("\nAVAILABLE SITES:")
    typer.echo("=" * 40)
    for idx, site in meta_df.iterrows():
        site_info = f"Site ID: {site['site_id']}"
        if 'location' in site:
            site_info += f" | Location: {site['location']}"
        if 'site_type' in site:
            site_info += f" | Type: {site['site_type']}"
        if 'description' in site:
            site_info += f" | {site['description'][:40]}"
        typer.echo(site_info)
    
    typer.echo(f"\nTotal sites: {len(meta_df)}")

@app.command()
def train_all(
    force_retrain: bool = typer.Option(True, "--force-retrain", help="Force retraining of models (default: True for this command)")
):
    # Train all models
    print(" Training all models...")
    
    ops_df = load_operations_data()
    meta_df = load_site_meta()
    df = merge_data(ops_df, meta_df)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    
    feature_cols = [
        'day_of_week', 'month', 'quarter', 'day_of_year', 'week_of_year',
        'units_produced_lag_1', 'units_produced_lag_7', 'units_produced_lag_14', 'units_produced_lag_30',
        'units_produced_roll_mean_7', 'units_produced_roll_mean_14', 'units_produced_roll_mean_30',
        'power_kwh_lag_1', 'power_kwh_lag_7', 'power_kwh_lag_14', 'power_kwh_lag_30',
        'power_kwh_roll_mean_7', 'power_kwh_roll_mean_14', 'power_kwh_roll_mean_30',
        'efficiency', 'downtime_ratio', 'temperature_c', 'rainfall_mm', 'is_holiday'
    ]
    
    print("Training units models...")
    models_units = train_models(df_features, 'units_produced', feature_cols, force_retrain=force_retrain)
    print("Training power models...")
    models_power = train_models(df_features, 'power_kwh', feature_cols, force_retrain=force_retrain)
    
    print("Detecting anomalies...")
    df_anomalies = detect_anomalies_zscore(df_features, 'downtime_minutes', 3.0)
    alerts_df = generate_alerts(df_anomalies)
    
    sites = list(models_units.keys())
    typer.echo(f"\nTRAINING COMPLETE!")
    typer.echo(f"Trained models for {len(sites)} sites")
    typer.echo(f"Targets: units_produced, power_kwh")
    typer.echo(f"Generated {len(alerts_df)} anomaly alerts")
    typer.echo(f"Models saved to models/ directory")
    
    for site in sites:
        status = " (from cache)" if models_units[site].get('loaded_from_persistence', False) else " (freshly trained)"
        typer.echo(f"    Site {site}{status}")
    
    return df_features, models_units, models_power, alerts_df, feature_cols


@app.command()
def health():
    # Check pipeline status
    try:
        ops_df = load_operations_data()
        meta_df = load_site_meta()
        
        typer.echo("PIPELINE HEALTH CHECK")
        typer.echo("=" * 40)
        typer.echo(f"Operations data: {ops_df.shape[0]:,} rows x {ops_df.shape[1]} columns")
        typer.echo(f"Site metadata: {len(meta_df)} sites")
        typer.echo(f"Date range: {ops_df['date'].min().strftime('%Y-%m-%d')} to {ops_df['date'].max().strftime('%Y-%m-%d')}")
        typer.echo(f"Targets available: {list(ops_df.columns)}")
        
        import os
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            typer.echo(f"Model cache: {len(model_files)} files in {models_dir}/")
        else:
            typer.echo("Model cache: No models directory found")
            
        typer.echo("\nPipeline is healthy and ready for queries!")
        
    except Exception as e:
        typer.echo(f"Pipeline health check failed: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()