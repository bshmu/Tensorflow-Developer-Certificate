"""
Skills Covered:
1) Tensorflow datasets
2) Bidirectional LSTM for NLP
3) CNN for NLP
4) GRU for NLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Week 3
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print('padded [1]:', padded[1])
print('padded [1] decoded:', decode_review(padded[1]))
print('training sentences [1]:', training_sentences[1])
print('training labels final:', training_labels_final)

# Multiple layer LSTM -- make sure all but last LTSM layer returns a sequence
model_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()

# Conv1D with GlobalAveragePooling layer
model_conv1d = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_conv1d.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_conv1d.summary()

# GRU
model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gru.summary()

model_l = ['lstm', 'gru', 'conv1d']
history_l = []
for model in [model_lstm, model_gru, model_conv1d]:
    history = model.fit(padded,
                        training_labels_final,
                        epochs=10,
                        validation_data=(testing_padded, testing_labels_final))
    history_l.append(history)

# Plot the results
def plot_graphs(history_l, model_l, string):
    # Training set
    leg = []
    for history, model in zip(history_l, model_l):
        plt.plot(history.history[string])
        leg.append(model + '_train')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(leg)
    plt.show()

    # Validation set
    leg2 = []
    for history, model in zip(history_l, model_l):
        plt.plot(history.history['val_' + string])
        leg2.append(model + '_val')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(leg2)
    plt.show()

plot_graphs(history_l, model_l, 'accuracy')
plot_graphs(history_l, model_l, 'loss')