"""
Skills Covered:
1) IMBD dataset in tensorflow_datasets
2) Adding embeddings layer to NLP model -- the weights are of shape (vocab_size, embedding_dim)
tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)
3) Getting data from csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
Concepts:
Neural network embeddings are low-dimensional, learned continuous vector representations of discrete variables.

Purposes of neural network embeddings:
1) Finding nearest neighbors in the embedding space.
2) As input to a machine learning model for a supervised task.
3) Visualization of concepts and relations between categories.

Thus, for each word in the BBC text we're learning a 16-dim vector called an "embedding".
This vector is grouping similar words together (e.g., taking a cosine similarity of each vector for words like 
"investors" and "profits" is > 95%!), and we can use this vector to make predictions!

"""


# Week 2
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

# Tokenize words and pad the sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
# Reverse the word index -- get numbers as keys and text as values
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode padded text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(padded[3]))
print(training_sentences[3])

# DNN with word embedding layer (16 embeddings)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

# Binary cross-entropy loss since we are predicting a binary class
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

words_l = []
embeddings_l = []
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    words_l.append(word)
    embeddings_l.append(embeddings)

word_df = pd.DataFrame({'word': words_l, 'embeddings': embeddings_l})

# Week 2 Assignment
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_portion = .8

sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

# Get words from BBC csv
text_df = pd.read_csv(r'C:\Users\User\repos\tfdev\data\bbc-text.csv')
for text_label, text_row in zip(text_df['category'], text_df['text']):
    labels.append(text_label)
    sentence = []
    for word in text_row.split(' '):
        if word not in stopwords and word != '':
            sentence.append(word)
    sentences.append(sentence)

# Train and validation sets
train_size = int(len(sentences) * training_portion)
train_sentences = sentences[:train_size] 
train_labels = labels[:train_size]
validation_sentences = sentences[train_size:] 
validation_labels = labels[train_size:]

# Tokenize words and pad sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) 
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index 
train_sequences = tokenizer.texts_to_sequences(train_sentences) 
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences) 
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
label_tokenizer = Tokenizer() 
label_tokenizer.fit_on_texts(labels)
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels)) 
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Create model with average pooling
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
history = model.fit(train_padded, training_label_seq, epochs=10, validation_data=(validation_padded, validation_label_seq), verbose=2)

# Plot the accuracy/lost
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Get the words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
weights = model.layers[0].get_weights()[0]
words_l = []
embeddings_l = []
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    words_l.append(word)
    embeddings_l.append(embeddings)

word_df = pd.DataFrame({'word': words_l, 'embeddings': embeddings_l})

