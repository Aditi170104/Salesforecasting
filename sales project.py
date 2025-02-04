import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Load dataset
df = pd.read_csv("Walmart.csv")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Aggregate sales by month
df.set_index('Date', inplace=True)
monthly_sales = df['Weekly_Sales'].resample('M').sum()

# Plot sales data
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Monthly Sales', marker='o')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Sales Over Time')
plt.legend()
plt.grid()
plt.show()

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposition (Trend, Seasonality, and Residuals)
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(monthly_sales, label='Original', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='purple')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Fit ARIMA model
model = ARIMA(monthly_sales, order=(5, 1, 0))  # (p,d,q) parameters
df_arima = model.fit()

# Forecast next 12 months
forecast = df_arima.forecast(steps=12)

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, label='Observed Sales', marker='o')
plt.plot(forecast, label='Forecast', linestyle='dashed', color='red', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast for Next 12 Months')
plt.legend()
plt.grid()
plt.show()

# 🚀 FIX: Drop NaN values before plotting ACF
residual_cleaned = residual.dropna()  # Remove NaN values

# Ensure we don't use more lags than available data points
max_lags = min(40, len(residual_cleaned) - 1)  

# Additional analysis of residuals (autocorrelation)
plt.figure(figsize=(10, 6))
plot_acf(residual_cleaned, lags=max_lags)  # Adjust lags dynamically
plt.title('Autocorrelation of Residuals')
plt.show()

# Check the mean and standard deviation of residuals
residual_mean = residual_cleaned.mean()
residual_std = residual_cleaned.std()
print(f'Residual Mean: {residual_mean}')
print(f'Residual Standard Deviation: {residual_std}')
