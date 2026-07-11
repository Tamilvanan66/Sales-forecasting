"""
Sales Forecasting - XGBoost Regressor
=====================================
A complete ML pipeline for sales forecasting with:
- Time-based train/test split
- Lag feature engineering
- XGBoost regression
- RMSE and MAE evaluation
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_sales_data(n_days=1095, random_state=42):
    """Generate realistic synthetic sales data (3 years of daily data)."""
    np.random.seed(random_state)
    
    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='D')
    
    # Base trend
    trend = np.linspace(1000, 2500, n_days)
    
    # Seasonality (yearly)
    yearly = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Weekly pattern
    weekly = 150 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Monthly pattern
    monthly = 80 * np.sin(2 * np.pi * np.arange(n_days) / 30.44)
    
    # Holiday effects (approximate)
    month = np.array([d.month for d in dates])
    day = np.array([d.day for d in dates])
    
    # Black Friday / Cyber Monday boost (November)
    black_friday = np.where(
        (month == 11) & (day >= 23) & (day <= 30), 800, 0
    )
    
    # Christmas boost (December 15-25)
    christmas = np.where(
        (month == 12) & (day >= 15) & (day <= 25), 600, 0
    )
    
    # Summer boost (June-August)
    summer = np.where((month >= 6) & (month <= 8), 100, 0)
    
    # Random noise
    noise = np.random.normal(0, 50, n_days)
    
    # Combine all components
    sales = trend + yearly + weekly + monthly + black_friday + christmas + summer + noise
    sales = np.maximum(sales, 100).round(2)  # Minimum sales of 100
    
    # Additional features
    day_of_week = np.array([d.weekday() for d in dates])
    week_of_year = np.array([d.isocalendar()[1] for d in dates])
    quarter = np.array([d.quarter for d in dates])
    
    data = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': day_of_week,
        'week_of_year': week_of_year,
        'month': month,
        'quarter': quarter,
        'year': np.array([d.year for d in dates]),
        'is_weekend': (day_of_week >= 5).astype(int),
        'is_month_start': np.array([d.is_month_start for d in dates]).astype(int),
        'is_month_end': np.array([d.is_month_end for d in dates]).astype(int),
        'is_quarter_start': np.array([d.is_quarter_start for d in dates]).astype(int),
        'is_quarter_end': np.array([d.is_quarter_end for d in dates]).astype(int),
    })
    
    return data


def create_lag_features(df, target_col='sales', lags=[1, 2, 3, 5, 7, 14, 21, 30]):
    """Create lag features for time series forecasting."""
    print("\n  Creating lag features...")
    
    df_lagged = df.copy()
    
    for lag in lags:
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
        print(f"    Added lag_{lag}")
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df_lagged[f'rolling_mean_{window}'] = df_lagged[target_col].shift(1).rolling(window=window).mean()
        df_lagged[f'rolling_std_{window}'] = df_lagged[target_col].shift(1).rolling(window=window).std()
        df_lagged[f'rolling_min_{window}'] = df_lagged[target_col].shift(1).rolling(window=window).min()
        df_lagged[f'rolling_max_{window}'] = df_lagged[target_col].shift(1).rolling(window=window).max()
        print(f"    Added rolling_{window} (mean, std, min, max)")
    
    # Expanding statistics
    df_lagged['expanding_mean'] = df_lagged[target_col].shift(1).expanding().mean()
    print(f"    Added expanding_mean")
    
    # Day-over-day change
    df_lagged['daily_change'] = df_lagged[target_col].diff(1)
    df_lagged['daily_pct_change'] = df_lagged[target_col].pct_change(1)
    print(f"    Added daily_change, daily_pct_change")
    
    # Drop rows with NaN from lagging
    df_lagged = df_lagged.dropna()
    
    print(f"\n  Rows after lag feature creation: {len(df_lagged)}")
    
    return df_lagged


def prepare_features(df):
    """Prepare feature matrix for model training."""
    feature_cols = [
        # Time features
        'day_of_week', 'week_of_year', 'month', 'quarter', 'year',
        'is_weekend', 'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end',
        
        # Lag features
        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
        
        # Rolling features
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
        'rolling_mean_30', 'rolling_std_30', 'rolling_min_30', 'rolling_max_30',
        
        # Other features
        'expanding_mean', 'daily_change', 'daily_pct_change',
    ]
    
    X = df[feature_cols]
    y = df['sales']
    
    return X, y, feature_cols


def time_based_split(df, test_ratio=0.2):
    """Time-based train/test split (no shuffling)."""
    split_idx = int(len(df) * (1 - test_ratio))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model."""
    print("\n" + "=" * 60)
    print("XGBOOST MODEL TRAINING")
    print("=" * 60)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0,
    }
    
    model = xgb.XGBRegressor(**params)
    
    eval_set = [(X_train, y_train)]
    if X_val is not None:
        eval_set.append((X_val, y_val))
    
    print("\n  Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    
    print(f"  Model trained with {model.n_estimators} trees")
    
    return model


def evaluate_model(model, X_test, y_test, feature_cols):
    """Evaluate model with RMSE and MAE."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    
    # Core metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100
    
    print(f"\n  {'Metric':<25} {'Value':>12}")
    print(f"  {'-'*37}")
    print(f"  {'RMSE':<25} {rmse:>12.2f}")
    print(f"  {'MAE':<25} {mae:>12.2f}")
    print(f"  {'R² Score':<25} {r2:>12.4f}")
    print(f"  {'MAPE (%)':<25} {mape:>12.2f}%")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:.4f}")
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
    }
    
    return metrics, importance, y_pred


def plot_results(y_test, y_pred, importance, metrics, dates_test):
    """Generate and save visualization plots."""
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sales Forecasting - XGBoost Results Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted (Time Series)
    axes[0, 0].plot(dates_test.values, y_test.values, label='Actual', color='#3b82f6', linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(dates_test.values, y_pred, label='Predicted', color='#ef4444', linewidth=1.5, alpha=0.8)
    axes[0, 0].fill_between(dates_test.values, y_test.values, y_pred, alpha=0.1, color='#ef4444')
    axes[0, 0].set_title('Actual vs Predicted Sales', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Residuals Distribution
    residuals = y_test.values - y_pred
    axes[0, 1].hist(residuals, bins=30, color='#22c55e', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residual (Actual - Predicted)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Feature Importance (Top 15)
    top_features = importance.head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    axes[1, 0].barh(range(len(top_features)), top_features['importance'].values, color=colors)
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['feature'].values, fontsize=9)
    axes[1, 0].set_title('Top 15 Feature Importance', fontweight='bold')
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].invert_yaxis()
    
    # 4. Metrics Summary
    axes[1, 1].axis('off')
    metrics_text = [
        ['Metric', 'Value'],
        ['RMSE', f'{metrics["rmse"]:.2f}'],
        ['MAE', f'{metrics["mae"]:.2f}'],
        ['R² Score', f'{metrics["r2_score"]:.4f}'],
        ['MAPE (%)', f'{metrics["mape"]:.2f}%'],
    ]
    table = axes[1, 1].table(
        cellText=metrics_text[1:],
        colLabels=metrics_text[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    axes[1, 1].set_title('Model Metrics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'sales_forecast_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")
    
    # Additional: Forecast vs Actual scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', edgecolors='white', s=30)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Sales ($)', fontsize=12)
    ax.set_ylabel('Predicted Sales ($)', fontsize=12)
    ax.set_title('Actual vs Predicted Scatter Plot', fontweight='bold', fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    scatter_path = os.path.join(OUTPUT_DIR, 'scatter_plot.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {scatter_path}")
    
    # Additional: Monthly aggregated comparison
    test_df = pd.DataFrame({'date': dates_test.values, 'actual': y_test.values, 'predicted': y_pred})
    test_df['date'] = pd.to_datetime(test_df['date'])
    monthly = test_df.groupby(test_df['date'].dt.to_period('M')).agg({'actual': 'sum', 'predicted': 'sum'}).reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly['date'], monthly['actual'], marker='o', label='Actual', color='#3b82f6', linewidth=2)
    ax.plot(monthly['date'], monthly['predicted'], marker='s', label='Predicted', color='#ef4444', linewidth=2)
    ax.fill_between(monthly['date'], monthly['actual'], monthly['predicted'], alpha=0.15, color='#ef4444')
    ax.set_title('Monthly Sales: Actual vs Predicted', fontweight='bold', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales ($)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    monthly_path = os.path.join(OUTPUT_DIR, 'monthly_comparison.png')
    plt.savefig(monthly_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {monthly_path}")


def main():
    """Main pipeline."""
    print("\n" + "=" * 60)
    print("  SALES FORECASTING PIPELINE")
    print("  Model: XGBoost | Metrics: RMSE, MAE")
    print("=" * 60)
    
    # Step 1: Generate/Load data
    print("\n[1/7] Generating synthetic sales data...")
    df = generate_sales_data(n_days=1095, random_state=42)
    data_path = os.path.join(DATA_DIR, 'sales_data.csv')
    df.to_csv(data_path, index=False)
    print(f"  Dataset saved: {data_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Sales range: ${df['sales'].min():.2f} to ${df['sales'].max():.2f}")
    
    # Step 2: Create lag features
    print("\n[2/7] Creating lag features...")
    df_lagged = create_lag_features(df)
    
    # Step 3: Prepare features
    print("\n[3/7] Preparing features...")
    X, y, feature_cols = prepare_features(df_lagged)
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    
    # Step 4: Time-based split
    print("\n[4/7] Splitting data (80/20 time-based)...")
    split_idx = int(len(df_lagged) * 0.8)
    
    train_df = df_lagged.iloc[:split_idx]
    test_df = df_lagged.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    X_test = test_df[feature_cols]
    y_test = test_df['sales']
    dates_test = test_df['date']
    
    print(f"  Train: {len(X_train)} samples (until {train_df['date'].max()})")
    print(f"  Test:  {len(X_test)} samples (from {test_df['date'].min()})")
    
    # Step 5: Train XGBoost
    print("\n[5/7] Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluate
    print("\n[6/7] Evaluating model...")
    metrics, importance, y_pred = evaluate_model(model, X_test, y_test, feature_cols)
    
    # Step 7: Plot
    print("\n[7/7] Generating visualizations...")
    plot_results(y_test, y_pred, importance, metrics, dates_test)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'xgboost_sales_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n  Model saved: {model_path}")
    
    # Save feature importance
    importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'), index=False)
    
    # Save predictions
    predictions = pd.DataFrame({
        'date': dates_test.values,
        'actual': y_test.values,
        'predicted': y_pred,
        'residual': y_test.values - y_pred,
    })
    predictions.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2_score']:.4f}")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
