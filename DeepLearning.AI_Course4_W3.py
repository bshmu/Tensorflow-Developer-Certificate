"""
Skills Covered:
1) Windowed dataset
2) Lambda layers in sequential (used for transformations)
3) Dynamic learning rate adjustment
4) LSTM model applied to time series
5) Forecasting

Data input pipeline concept:
tf.data allows you to build input pipelines. For example, the pipeline for a text model might extract symbols from raw
text data, convert them to embeddings with a lookup table, and then batch together sequences of different lengths.

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

# Create the series
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)  # Update with noise

# Split data
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# Model 1 (RNN) - search for best learning rate that minimizes loss
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)
# The batches from windowed dataset need to have an additional dimension to be fed into the sequential model
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model1.compile(loss=tf.keras.losses.Huber(),
               optimizer=optimizer1,
               metrics=["mae"])
history = model1.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])

# Model 2 (RNN) - implement learning rate from previous model
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
dataset = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)
# The batches from windowed dataset need to have an additional dimension to be fed into the sequential model
model2 = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

optimizer2 = tf.keras.optimizers.SGD(learning_rate=5e-5, momentum=0.9)
model2.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer2,
              metrics=["mae"])
history = model2.fit(dataset,epochs=400)

# Forecasts for model 2
forecast2 = []
for time in range(len(series) - window_size):
    forecast2.append(model2.predict(series[time:time + window_size][np.newaxis]))
forecast2 = forecast2[split_time-window_size:]
results2 = np.array(forecast2)[:, 0, 0]

# Plot of model 2
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results2)
print('MAE (RNN):', tf.keras.metrics.mean_absolute_error(x_valid, results2).numpy())

# Model 3 (LSTM) - search for best learning rate that minimizes loss
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# The batches from windowed dataset need to have an additional dimension to be fed into the sequential model
model3 = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer3 = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model3.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer3,
              metrics=["mae"])
history = model3.fit(dataset, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])

# Model 4 (LSTM) - implement learning rate from previous model
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# The batches from windowed dataset need to have an additional dimension to be fed into the sequential model
model4 = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])
model4.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),metrics=["mae"])
history = model4.fit(dataset, epochs=500, verbose=0)

# Forecasts for model 4
forecast4 = []
results4 = []
for time in range(len(series) - window_size):
    forecast4.append(model4.predict(series[time:time + window_size][np.newaxis]))
forecast4 = forecast4[split_time-window_size:]
results4 = np.array(forecast4)[:, 0, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results4)
print('MAE (LSTM):', tf.keras.metrics.mean_absolute_error(x_valid, results2).numpy())

