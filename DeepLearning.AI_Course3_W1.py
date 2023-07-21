"""
Skills Covered:
1) Tokenizer object -- Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    a) tokenizer.fit_on_texts()
    b) tokenizer.word_index
    c) tokenizer.text_to_sequences()
2) Pad Sequences object
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
Concepts:
1) Word Tokenizer object
    a) fit_on_texts method
    b) text_to_sequences method
2) Pad Sequences object
"""

# Week 1

# Create tokenizer object and fit on list on sentences
sentences = ['i love my dog', 'I, love my cat', 'You love my dog!']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

# tokenizer.word_index is a dict with words as keys and token as the value
word_index = tokenizer.word_index
print("\nWord Index = ", word_index)

# tokenizer.text_to_sequences is a list of list of ints corresponding to each word
sequences = tokenizer.texts_to_sequences(sentences)
print("\nSequences = ", sequences)

# pad_sequences forms a 2D numpy array padded with zeros (all inner arrays the same len)
# Can set padding to "pre" or "post"
padded = pad_sequences(sequences, maxlen=5)
print("\nPadded Sequences:\n", padded)

# Try with words that the tokenizer wasn't fit to
test_data = ['i really love my dog', 'my dog loves my manatee']
test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)
padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)

# Week 1 Assignment
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves" ]
sentences = []
labels = []

text_df = pd.read_csv(r'C:\Users\User\repos\tfdev\data\bbc-text.csv')
for text_label, text_row in zip(text_df['category'], text_df['text']):
    labels.append(text_label)
    sentence = []
    for word in text_row.split(' '):
        if word not in stopwords and word != '':
            sentence.append(word)
    sentences.append(sentence)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # Matches all words in the sentence with a token and include out of vocabulary token
sequences = tokenizer.texts_to_sequences(sentences)  # Transforms to list of lists of ints corresponding to each word
padded = pad_sequences(sequences, padding='post')  # Convert to 2D numpy padded with zeros in front

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)