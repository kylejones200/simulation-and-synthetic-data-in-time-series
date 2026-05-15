# Description: Short example for Simulation and synthetic data in Time Series.



# Simulate random noise

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

white_noise = np.random.normal(loc=0, scale=1, size=500)
# Plot white noise
plt.figure(figsize=(10, 6))
plt.plot(white_noise, label='Random Noise')
plt.title('Simulated Random Noise')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('random_noise_simulation.png')
plt.show()


# Simulate AR(1) process
ar_params = [1, -0.7]  # AR(1) with phi=0.7
ma_params = [1]  # No MA component
ar_process = ArmaProcess(ar_params, ma_params)
ar_data = ar_process.generate_sample(nsample=500)
# Plot AR(1) process
plt.figure(figsize=(10, 6))
plt.plot(ar_data, label='AR(1) Process')
plt.title('Simulated AR(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('ar1_simulation.png')
plt.show()

# Simulate MA(1) process
ar_params = [1]  # No AR component
ma_params = [1, 0.5]  # MA(1) with theta=0.5
ma_process = ArmaProcess(ar_params, ma_params)
ma_data = ma_process.generate_sample(nsample=500)

# Plot MA(1) process
plt.figure(figsize=(10, 6))
plt.plot(ma_data, label='MA(1) Process')
plt.title('Simulated MA(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('ma1_simulation.png')
plt.show()


# Simulate ARIMA(1,1,1) process
ar_params = [1, -0.5]  # AR(1) with phi=0.5
ma_params = [1, 0.4]   # MA(1) with theta=0.4
arima_process = ArmaProcess(ar_params, ma_params)
arima_data = np.cumsum(arima_process.generate_sample(nsample=500))  # Differencing (I)
# Plot ARIMA(1,1,1) process
plt.figure(figsize=(10, 6))
plt.plot(arima_data, label='ARIMA(1,1,1) Process')
plt.title('Simulated ARIMA(1,1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('arima_simulation.png')
plt.show()

# Simulate a non-stationary time series with trend
time = np.arange(500)
trend_series = 0.05 * time + np.random.normal(loc=0, scale=1, size=500)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(trend_series, label='Non-Stationary Series with Trend')
plt.title('Simulated Time Series with Trend')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('trend_simulation.png')
plt.show()

"""
Evaluating ARIMA on Simulated Data
"""


# Fit ARIMA(1,1,0) to the simulated data
model = ARIMA(trend_series, order=(1, 1, 0))
fit = model.fit()
# Plot the original series and model predictions
plt.figure(figsize=(10, 6))
plt.plot(trend_series, label='Original Data')
plt.plot(fit.fittedvalues, label='Fitted ARIMA Model', linestyle='--')
plt.title('ARIMA Model on Simulated Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('arima_model_simulation.png')
plt.show()
