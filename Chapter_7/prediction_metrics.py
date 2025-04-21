import pandas as pd
import numpy as np

# Test on the following sample data
# 
np.random.seed(42)

# Simulate stock predictions and actuals
days = 100
predicted_prices = np.cumsum(np.random.randn(days) * 2 + 100)
actual_prices = predicted_prices + np.random.randn(days) * 3

# Simulate portfolio and benchmark returns
portfolio_returns = np.random.normal(loc=0.001, scale=0.01, size=days)
benchmark_returns = np.random.normal(loc=0.0005, scale=0.008, size=days)

# Simulate FINGPT alerts and user actions
alerts = pd.DataFrame({
    "date": pd.date_range(start="2024-01-01", periods=days),
    "alert_type": np.random.choice(["positive", "negative"], size=days),
    "was_actioned": np.random.choice([True, False], size=days, p=[0.7, 0.3]),
    "is_successful": np.random.choice([True, False], size=days, p=[0.6, 0.4]),
    "time_to_action": np.random.randint(0, 4, size=days)  # in days
})

# ---------- KPI Calculations ---------- #

# Prediction Accuracy
directional_accuracy = np.mean(
    np.sign(np.diff(predicted_prices)) == np.sign(np.diff(actual_prices))
)

mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))

# Alpha and Outperformance Rate
excess_returns = portfolio_returns - benchmark_returns
alpha = np.mean(excess_returns)
outperformance_rate = np.mean(portfolio_returns > benchmark_returns)

# Timeliness and Responsiveness
significant_moves = np.random.choice([True, False], size=days, p=[0.3, 0.7])
early_opportunities = alerts[(alerts.alert_type == "positive") & alerts.was_actioned & alerts.is_successful & significant_moves]
risk_mitigated = alerts[(alerts.alert_type == "negative") & alerts.was_actioned & alerts.is_successful & significant_moves]

early_opportunity_capture_rate = len(early_opportunities) / sum(significant_moves)
risk_mitigation_effectiveness = len(risk_mitigated) / sum(significant_moves)
alert_utilisation_rate = np.mean(alerts.was_actioned)

# Risk Management
portfolio_volatility = np.std(portfolio_returns)
max_drawdown = np.max(np.maximum.accumulate(np.cumsum(portfolio_returns)) - np.cumsum(portfolio_returns))
sharpe_ratio = np.mean(portfolio_returns - 0.0001) / np.std(portfolio_returns)

# Scenario Analysis Effectiveness (simulated as qualitative)
scenario_analysis_effectiveness = "Effective in 3 out of 4 market shocks"

# Client Satisfaction
client_retention_rate = 0.93
net_asset_growth = 1.15  # 15% increase

# Improvement in FINGPT Accuracy
pre_mape = mape * 1.2  # simulate worse before FINGPT
post_mape = mape
pre_directional_accuracy = directional_accuracy - 0.1
post_directional_accuracy = directional_accuracy

# Risk-adjusted impact of FINGPT
drawdown_with_fingpt = max_drawdown * 0.8  # improved
drawdown_without_fingpt = max_drawdown

# System Performance (simulated)
system_performance = {
    "uptime_percentage": 99.8,
    "data_latency_seconds": 2,
    "alert_precision": 0.78,
    "alert_recall": 0.81,
    "user_engagement_rate": 0.85
}

# Compile all KPIs
kpi_summary = {
    "Directional Accuracy": f"{directional_accuracy:.2%}",
    "MAPE": f"{mape:.2f}%",
    "RMSE": f"{rmse:.2f}",
    "Alpha (Excess Return)": f"{alpha:.4f}",
    "Outperformance Rate": f"{outperformance_rate:.2%}",
    "Early Opportunity Capture Rate": f"{early_opportunity_capture_rate:.2%}",
    "Risk Mitigation Effectiveness": f"{risk_mitigation_effectiveness:.2%}",
    "Alert Utilisation Rate": f"{alert_utilisation_rate:.2%}",
    "Portfolio Volatility": f"{portfolio_volatility:.4f}",
    "Maximum Drawdown": f"{max_drawdown:.4f}",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
    "Scenario Analysis Effectiveness": scenario_analysis_effectiveness,
    "Client Retention Rate": f"{client_retention_rate:.2%}",
    "Net Asset Growth": f"{net_asset_growth:.2f}x",
    "Pre-FINGPT MAPE": f"{pre_mape:.2f}%",
    "Post-FINGPT MAPE": f"{post_mape:.2f}%",
    "Pre-FINGPT Directional Accuracy": f"{pre_directional_accuracy:.2%}",
    "Post-FINGPT Directional Accuracy": f"{post_directional_accuracy:.2%}",
    "Drawdown With FINGPT": f"{drawdown_with_fingpt:.4f}",
    "Drawdown Without FINGPT": f"{drawdown_without_fingpt:.4f}",
    "System Uptime": f"{system_performance['uptime_percentage']}%",
    "Data Latency": f"{system_performance['data_latency_seconds']} sec",
    "Alert Precision": f"{system_performance['alert_precision']:.2%}",
    "Alert Recall": f"{system_performance['alert_recall']:.2%}",
    "User Engagement Rate": f"{system_performance['user_engagement_rate']:.2%}"
}

kpi_df = pd.DataFrame(list(kpi_summary.items()), columns=["KPI", "Value"])

kpi_df.to_csv('Data/kpi_metrics.csv')