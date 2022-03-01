import os

import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Flatten, Add, concatenate, Embedding
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#modified from https://towardsdatascience.com/a-complete-step-by-step-tutorial-on-sentiment-analysis-in-keras-and-tensorflow-ea420cc8913f

#begin data processing
#create pandas struct
df = pd.DataFrame()
#load data set
df = pd.read_csv('data/archive/amazon_baby.csv')
print(df)
#adding sentiment columnn to data set.
#rating is 0 to 5 stars 3+ will be positive else negative
df['sentiments'] = df.rating.apply(lambda x: 0 if x in [1, 2] else 1)
print(df)
#initialize tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
split = round(len(df)*0.8)
train_reviews = df['review'][:split]
train_label = df['sentiments'][:split]
test_reviews = df['review'][split:]
test_label = df['sentiments'][split:]
#make sure its all strings
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for row in train_reviews:
    training_sentences.append(str(row))
for row in train_label:
    training_labels.append(row)
for row in test_reviews:
    testing_sentences.append(str(row))
for row in test_label:
    testing_labels.append(row)

'''Here, vocab_size 40,000 means we will take 40,000 unique 
words to train the network. Embedding dimension 16 means each 
word will be represented by a 16-dimensional vector. 
Max_length 120 represents the length of each review. We will keep 
120 words from each review. If originally the comment is longer 
than 120 words, it will be truncated. The term trunc_type is set 
to be ‘post’. So, the review will be truncated at the end when a 
review is bigger than 120 words. On the other hand, if the review 
is less than 120 words it will be padded to make 120 words.
In the end, padding_type ‘post’ means padding will be applied at
the end, not in the beginning.'''

vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'

#tokenize data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
print(word_index)


sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)

#end data processing

#begin defining model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,       input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print the model
model.summary()
model.save('model/test_')


#tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#train 
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
num_epochs = 20
history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))


#plot training and validation

#matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and validation loss')
plt.figure()