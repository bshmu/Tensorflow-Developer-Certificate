"""
Skills Covered:
1) Basic time series analysis
2) Mean absolute error
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    seasonal_pattern = np.where(season_time < 0.1, np.cos(season_time * 7 * np.pi), 1 / np.exp(5 * season_time))
    return amplitude * seasonal_pattern

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def moving_average_forecast(series, window_size):
    """
    Forecasts the mean of the last few values.
    If window_size=1, then this is equivalent to naive forecast
    """
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

# Series
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)  # Create the series
series += noise(time, noise_level, seed=42)  # Update with noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title('Base series')
plt.show()

# Split for forecasting
train_pct = 0.75
split_time = int(np.round(train_pct*len(time)))
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.title('train series')
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.title('validation series')
plt.show()

# Naive forecasting
naive_forecast = series[split_time - 1:-1]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)

print('mse (naive):', keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print('mae (naive):', keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

# Simple moving average
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print('mse (sma):', keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print('mae (sma):', keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())

# Differenced series
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.title('Differenced series')
plt.show()

# SMA on differenced series
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

# SMA on differenced series plus trend and seasonality (add back the past values)
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg 

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print('mse (sma + past, differenced):', keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print('mae (sma + past, differenced):', keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

# Smoothed past values
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg 
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print('mse (sma + smoothed past, differenced):', keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print('mae (sma + smoothed past, differenced):', keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
