import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def windowed_dataset(series, batch_size, n_past, n_future, shift):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

def normalize_series(data, min, max):
    return ((data - min) / max)

def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

df = pd.read_csv(r'C:\Users\User\repos\tfdev\data\Weekly_U.S._No_2_Diesel_Retail_Prices.csv')

data = df['$/gal'].values

split = 0.8
split_time = int(len(data) * split)
x_train = data[:split_time]
x_valid = data[split_time:]

tf.keras.backend.clear_session()
tf.random.set_seed(42)

batch_size = 32
n_past = 10
n_future = 10
shift = 1

train_set = windowed_dataset(x_train, batch_size, n_past, n_future, shift)
valid_set = windowed_dataset(x_valid, batch_size, n_past, n_future, shift)

model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.Dense(1)])

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9), metrics=['mae'])
history = model.fit(train_set, epochs=10)

rnn_forecast = model_forecast(model, data, n_past, batch_size)
rnn_forecast = rnn_forecast[split_time - n_past:-1, 0]
x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
print('MAE:', result)

time = np.arange(1, len(data))
time_valid = time[split_time+10-1:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()
