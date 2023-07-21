"""
Skills Covered:
1) Dataset methods
    1) dataset.window(x) -- subset of elements of the input dataset of len x
    2) dataset.map(lambda x: ...) -- transformation on the dataset
    3) dataset.shuffle(x) -- datasets fills a buffer with x elements, and randomly samples from this buffer
    4) datset.flat_map(lambda x: x.batch(y)) -- flatten a dataset of windows into a single dataset, each batch has len y
    5) dataset.batch(y).prefetch(z) -- usual end to data pipeline, allows later elements to be prepared while current element is being processed
    6) dataset.from_tensor_slices() -- s
2) Passing the dataset object into keras model.fit()
3) Adjusting the learning rate dynamically
4) Forecasting

Data input pipeline concept:
tf.data allows you to build input pipelines. For example, the pipeline for a text model might extract symbols from raw
text data, convert them to embeddings with a lookup table, and then batch together sequences of different lengths.


"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset methods
# Create dataset object -- it is a chained iterator so you can sequentially call methods on it
dataset = tf.data.Dataset.range(10)

# Define a window len, shift, and drop remainer.
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print('after flat map:', window.numpy())

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
    print('after shuffle:')
    print(x.numpy(), y.numpy())
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print('after batch/prefetch:')
    print("x = ", x.numpy())
    print("y = ", y.numpy())

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope):
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
    """
    This function takes a series and uses tf.data.Dataset.from_tensor_slices() to construct a dataset object from data
    in memory. Then, it creates ((len(series) - window_size - 1) // shift) + 1 windows of size window_size+1.
    These windows are then shuffled (not the window's elements), and the last element becomes a separate window.
    The shuffle_buffer parameter controls how many windows are shuffled. For example, if the dataset contains 1000
    windows, but the shuffle_buffer is only 100, then only the first 100 windows are shuffled.
    Finally, the batch method controls how many windows are in a given batch. For example, if the dataset contains 1000
    windows, and the batch_size is set to 100, then you will see 10 batches containing 100 windows each. Batch also
    shuffles the windows, and prefetch prepares later elements while the current element is being processed.

    num_windows = ((len(series) - window_size - 1) // shift) + 1

    Therefore, the final output will be:
    num_windows//batch_size batches of dimension (batch_size, window_size)
    + 1 batch of dimension (num_windows%match_size, window_size)

    """
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# Create series
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)  # Update with noise

# Create train/valid sets
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Run single layer NN
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# Call windowed_dataset function on the series
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# With batch_size = 32 and window_size = 20, the 1000-element time series becomes a dataset of 31 batches of (32, 20)
# windows and 1 batch of (20, 20) window

# Define the model -- make sure to set the window size as the input shape
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model1 = tf.keras.models.Sequential([l0])
model1.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
model1.fit(dataset, epochs=100, verbose=0)
print("Layer weights {}".format(l0.get_weights()))

# Generate forecasts
forecast1 = []
for time in range(len(series) - window_size):
    forecast1.append(model1.predict(series[time:time + window_size][np.newaxis]))
forecast1 = forecast1[split_time-window_size:]
results1 = np.array(forecast1)[:, 0, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results1)

mae1 = tf.keras.metrics.mean_absolute_error(x_valid, results1).numpy()
print('mae_1:', mae1)

# Run DNN
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# model2.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
# model2.fit(dataset, epochs=100, verbose=0)


# Use LearningRateScheduler to find optimal learning rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model2.compile(loss="mse", optimizer=optimizer)
history = model2.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)

# CHeck model output to see where loss is minimized -- appears to be around 10^-5
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
# plt.show()

# Update the optimizer and refit the model -- loss should decrease smoothly
optimizer_updated = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
model2.compile(loss="mse", optimizer=optimizer_updated)
history = model2.fit(dataset, epochs=500)
loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

# Plot all but the first 10
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
print(plot_loss)
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
plt.show()

# Generate forecasts by running the model across the entire dataset
# Call model.predict with the series, add an additional dimension with np.newaxis, and append to list
forecast2 = []
for time in range(len(series) - window_size):
    forecast2.append(model2.predict(series[time:time + window_size][np.newaxis]))
# Filter the dataset to the validation set
forecast2 = forecast2[split_time-window_size:]
# Transform list of 2D arrays into a single array
results = np.array(forecast2)[:, 0, 0]
# Plot the figure
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
mae2 = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
print('mae_2:', mae2)
# plt.show()