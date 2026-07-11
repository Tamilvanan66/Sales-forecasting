# Sales Forecasting

A complete ML pipeline for sales forecasting using **XGBoost Regressor** with time-based train/test split and lag features.

## Project Structure

```
02-sales-forecasting/
├── sales_forecasting.py         # Main pipeline script
├── requirements.txt             # Python dependencies
├── data/
│   └── sales_data.csv           # Generated dataset (3 years daily sales)
├── models/
│   └── xgboost_sales_model.joblib  # Trained XGBoost model
└── outputs/
    ├── sales_forecast_results.png   # Dashboard visualization
    ├── scatter_plot.png             # Actual vs Predicted scatter
    ├── monthly_comparison.png       # Monthly aggregated comparison
    ├── feature_importance.csv       # Feature importance rankings
    ├── model_metrics.csv            # Model performance metrics
    └── predictions.csv              # Full predictions with dates
```

## Features

### Time Features
- `day_of_week`: Day of the week (0-6)
- `week_of_year`: Week number (1-52)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `is_weekend`: Weekend flag
- `is_month_start`, `is_month_end`: Month boundary flags
- `is_quarter_start`, `is_quarter_end`: Quarter boundary flags

### Lag Features
- `lag_1` to `lag_30`: Sales values at previous time steps
- `rolling_mean_{7,14,30}`: Rolling mean over windows
- `rolling_std_{7,14,30}`: Rolling standard deviation
- `rolling_min_{7,14,30}`: Rolling minimum
- `rolling_max_{7,14,30}`: Rolling maximum
- `expanding_mean`: Expanding mean
- `daily_change`: Day-over-day change
- `daily_pct_change`: Day-over-day percentage change

### Model
- **Algorithm**: XGBoost Regressor
- **Hyperparameters**:
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 500
  - subsample: 0.8
  - colsample_bytree: 0.8

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python sales_forecasting.py
```

## Results

| Metric    | Value      |
|-----------|------------|
| RMSE      | 142.27     |
| MAE       | 64.73      |
| R² Score  | 0.7474     |
| MAPE      | 2.41%      |

### Top Features
1. lag_1 (previous day sales)
2. rolling_max_7
3. lag_7 (weekly lag)
4. rolling_min_7
5. lag_21

## Output

The pipeline generates:
- **Trained model**: Saved as `xgboost_sales_model.joblib`
- **Visualizations**: Time series plot, residuals, feature importance, scatter plot, monthly comparison
- **Predictions CSV**: Full predictions with dates for analysis
- **Metrics CSV**: For easy comparison and reporting
