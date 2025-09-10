import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
import joblib
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import ttest_rel

def baseline_forecast_arima(series: pd.Series, steps: int = 14) -> np.ndarray:
    # Simple ARIMA baseline
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def improved_forecast_xgb(model, df: pd.DataFrame, target: str, features: list, steps: int = 14) -> np.ndarray:
    # Multi-step XGB forecast (approximate lag updates)
    last_row = df.iloc[-1:][features].copy()
    forecasts = []
    for _ in range(steps):
        pred = model.predict(last_row)[0]
        forecasts.append(pred)
        last_row[target] = pred
        # Update lags (simple shift)
        for lag in [1,7,14,30]:
            if f'{target}_lag_{lag}' in features:
                last_row[f'{target}_lag_{lag}'] = pred if lag == 1 else last_row[f'{target}_lag_{lag-1}']
    return np.array(forecasts)

def train_models(df: pd.DataFrame, target: str, features: list, force_retrain: bool = False):
    # Train LR and XGB models per site with caching
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = df['date'].max().strftime('%Y%m%d')
    model_filename = f"{models_dir}/{target}_models_{timestamp}.pkl"
    metadata_filename = f"{models_dir}/{target}_metadata_{timestamp}.json"
    
    current_date = datetime.now()
    recent_threshold = current_date - timedelta(days=7)
    
    models = {}
    load_success = False
    
    if not force_retrain and os.path.exists(model_filename):
        try:
            loaded_models = joblib.load(model_filename)
            import json
            with open(metadata_filename, 'r') as f:
                metadata = json.load(f)
            model_date = datetime.strptime(metadata['training_date'][:10], '%Y-%m-%d')
            
            if model_date >= recent_threshold:
                print(f"Using cached {target} models from {model_filename}")
                models = loaded_models
                load_success = True
                for site in models:
                    models[site]['loaded_from_persistence'] = True
                    models[site]['training_date'] = model_date
            else:
                print("Cached models too old, retraining...")
        except Exception as e:
            print(f"Cache load failed: {e}")
    
    if not load_success:
        print(f"Training {target} models for {len(df['site_id'].unique())} sites")
        sites = df['site_id'].unique()
        tscv = TimeSeriesSplit(n_splits=5)
        
        for site in sites:
            site_df = df[df['site_id'] == site].copy().dropna()
            if len(site_df) < 50:
                continue  # Skip sites with too little data
            X = site_df[features]
            y = site_df[target]
            
            # Cross-validation scores
            lr_maes, xgb_maes = [], []
            mape_lr_folds, mape_xgb_folds = [], []
            fold_errors_lr, fold_errors_xgb = [], []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                mae_lr = mean_absolute_error(y_test, y_pred_lr)
                lr_maes.append(mae_lr)
                fold_errors_lr.append(mae_lr)
                if y_test.sum() > 0 and (y_test > 0).sum() > 0:
                    mape_lr_folds.append(mean_absolute_percentage_error(y_test, y_pred_lr))
                
                # XGBoost with random search
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [0, 0.1],
                    'min_child_weight': [1, 3],
                    'gamma': [0, 0.1]
                }
                xgb_base = xgb.XGBRegressor(objective='reg:squarederror')
                search = RandomizedSearchCV(xgb_base, param_distributions=param_dist, n_iter=10, cv=3,
                                          scoring='neg_mean_absolute_error', random_state=42)
                search.fit(X_train, y_train)
                xgb_model = search.best_estimator_
                y_pred_xgb = xgb_model.predict(X_test)
                mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
                xgb_maes.append(mae_xgb)
                fold_errors_xgb.append(mae_xgb)
                if y_test.sum() > 0 and (y_test > 0).sum() > 0:
                    mape_xgb_folds.append(mean_absolute_percentage_error(y_test, y_pred_xgb))
            
            # Calculate CV stats
            avg_mae_lr, std_mae_lr = np.mean(lr_maes), np.std(lr_maes)
            avg_mae_xgb, std_mae_xgb = np.mean(xgb_maes), np.std(xgb_maes)
            t_stat, p_value = ttest_rel(fold_errors_xgb, fold_errors_lr)
            improvement = ((avg_mae_lr - avg_mae_xgb) / avg_mae_lr) * 100 if avg_mae_lr > 0 else 0
            
            avg_mape_lr = np.mean(mape_lr_folds) if mape_lr_folds else np.nan
            avg_mape_xgb = np.mean(mape_xgb_folds) if mape_xgb_folds else np.nan
            
            # Final fold for multi-step eval
            final_train_idx, final_test_idx = list(tscv.split(X))[-1]
            X_final_train, X_final_test = X.iloc[final_train_idx], X.iloc[final_test_idx]
            y_final_train, y_final_test = y.iloc[final_train_idx], y.iloc[final_test_idx]
            
            val_split = int(len(X_final_train) * 0.8)
            X_train_final, X_val_final = X_final_train.iloc[:val_split], X_final_train.iloc[val_split:]
            y_train_final, y_val_final = y_final_train.iloc[:val_split], y_final_train.iloc[val_split:]
            
            # Retrain models
            lr_final = LinearRegression().fit(X_final_train, y_final_train)
            
            early_stop_params = search.best_params_.copy()
            early_stop_params.update({'early_stopping_rounds': 10, 'eval_metric': 'mae', 'verbose': False})
            xgb_final = xgb.XGBRegressor(**early_stop_params)
            xgb_final.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=False)
            
            # Multi-step MAE (last 14 test points)
            if len(y_final_test) >= 14:
                multi_mae_lr = mean_absolute_error(y_final_test.tail(14), lr_final.predict(X_final_test.tail(14)))
                multi_mae_xgb = mean_absolute_error(y_final_test.tail(14), xgb_final.predict(X_final_test.tail(14)))
            else:
                multi_mae_lr = multi_mae_xgb = avg_mae_xgb
            
            # Feature importance and selection (top 15)
            importances = dict(zip(features, xgb_final.feature_importances_))
            feature_importance_scores = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, imp in feature_importance_scores[:15]]
            top_features = feature_importance_scores[:10]
            
            # Retrain on selected features
            xgb_final_selected = xgb.XGBRegressor(**early_stop_params)
            xgb_final_selected.fit(X_train_final[selected_features], y_train_final,
                                 eval_set=[(X_val_final[selected_features], y_val_final)], verbose=False)
            
            lr_final_selected = LinearRegression().fit(X_final_train[selected_features], y_final_train)
            
            # Store model info
            models[site] = {
                'lr_model': lr_final,
                'xgb_model': xgb_final,
                'features': features,
                'avg_mae_lr': avg_mae_lr, 'std_mae_lr': std_mae_lr,
                'avg_mae_xgb': avg_mae_xgb, 'std_mae_xgb': std_mae_xgb,
                'avg_mape_lr': avg_mape_lr, 'avg_mape_xgb': avg_mape_xgb,
                'multi_mae_lr': multi_mae_lr, 'multi_mae_xgb': multi_mae_xgb,
                'improvement_pct': improvement,
                'p_value': p_value,
                'feature_importances': top_features,
                'early_stopping_used': True,
                'loaded_from_persistence': False,
                'training_date': current_date
            }
        
        # Save models and metadata
        print(f"Saving {target} models")
        joblib.dump(models, model_filename)
        
        metadata = {
            'target': target,
            'features': features,
            'sites': list(sites),
            'data_shape': df.shape,
            'data_timestamp': timestamp,
            'training_date': current_date.isoformat(),
            'data_max_date': df['date'].max().isoformat()
        }
        with open(metadata_filename, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    print(f"{target} models ready for {len(models)} sites")
    return models

def forecast_future(df: pd.DataFrame, models: dict, target: str, steps: int = 14):
    # Multi-step forecast per site with feature updates
    forecasts = {}
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=steps, freq='D')
    
    for site, model_dict in models.items():
        site_df = df[df['site_id'] == site].copy().sort_values('date')
        features = model_dict['features']
        
        # Average exogenous features by season
        avg_temp = site_df.groupby([site_df['date'].dt.month, site_df['date'].dt.dayofyear])['temperature_c'].mean().to_dict()
        avg_rain = site_df.groupby([site_df['date'].dt.month, site_df['date'].dt.dayofyear])['rainfall_mm'].mean().to_dict()
        avg_downtime_ratio = site_df['downtime_ratio'].mean()
        avg_efficiency = site_df['efficiency'].mean()
        
        hist_target = site_df[target].tail(30).values
        last_row = site_df.iloc[-1:][features].copy()
        if last_row.shape[1] == 0:
            last_row = pd.DataFrame(columns=features).fillna(0)
        else:
            last_row = last_row.reindex(columns=features, fill_value=0)
        
        preds = []
        pred_history = list(hist_target)
        
        for step, future_date in enumerate(future_dates):
            # Update date features
            last_row['day_of_week'] = future_date.dayofweek
            last_row['month'] = future_date.month
            last_row['quarter'] = future_date.quarter
            last_row['day_of_year'] = future_date.dayofyear
            last_row['week_of_year'] = future_date.isocalendar().week
            
            # Update weather/holiday features
            month_doy = (future_date.month, future_date.dayofyear)
            last_row['temperature_c'] = avg_temp.get(month_doy, site_df['temperature_c'].mean())
            last_row['rainfall_mm'] = avg_rain.get(month_doy, site_df['rainfall_mm'].mean())
            last_row['is_holiday'] = 1 if future_date.weekday() >= 5 or (future_date.month == 1 and future_date.day == 1) else 0
            last_row['downtime_ratio'] = avg_downtime_ratio
            if target == 'units_produced':
                last_row['efficiency'] = avg_efficiency
            
            # Ensemble prediction (70% XGB, 30% LR)
            xgb_pred = model_dict['xgb_model'].predict(last_row)[0]
            lr_pred = model_dict['lr_model'].predict(last_row)[0]
            pred = 0.7 * xgb_pred + 0.3 * lr_pred
            preds.append(pred)
            pred_history.append(pred)
            
            # Update lags
            for lag in [1, 7, 14, 30]:
                lag_feat = f'{target}_lag_{lag}'
                if lag_feat in features:
                    if len(pred_history) >= lag:
                        last_row[lag_feat] = pred_history[-lag]
                    else:
                        last_row[lag_feat] = pred_history[0]
            
            # Update rolling means
            for window in [7, 14, 30]:
                roll_feat = f'{target}_roll_mean_{window}'
                if roll_feat in features:
                    if len(pred_history) >= window:
                        last_row[roll_feat] = np.mean(pred_history[-window:])
                    else:
                        last_row[roll_feat] = np.mean(pred_history)
        
        forecasts[site] = preds
    return forecasts