import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

import json
import os
import gensim
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np


# model hyper parameters
EMBEDDING_DIM =200
SEQUENCE_LENGTH_PERCENTILE = 90
n_layers = 2
hidden_units = 500
batch_size = 100
pretrained_embedding = False
# if we have pre-trained embeddings, specify if they are static or non-static embeddings
TRAINABLE_EMBEDDINGS = True
patience = 2
dropout_rate = 0.3
n_filters = 100
window_size = 8
dense_activation = "relu"
l2_penalty = 0.0003
epochs = 10
VALIDATION_SPLIT = 0.2
dictpath = './yelp.dict'
model_file = './models/yelp_cnnmodel0.model'
datapath = './data'
readlim = 500000


def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))



# reading in text from file
txtfile = open(os.path.join(datapath,'review.json'), mode='r', encoding='utf8')

txtlist = [json.loads(next(txtfile)) for line in range(readlim)]
txtfile.close()

# normalizing and tokenizing
tokens = [t['text'].strip(',.&').lower().split() for t in txtlist]
tokens = list(map(lambda y: list(filter(lambda x: x.isalpha(), y)), tokens))
labels = [t['stars'] for t in txtlist]
labels = keras.utils.to_categorical(labels)

# creating or loading dictionary
if dictpath == None:
    dictionary = gensim.corpora.Dictionary(tokens)
    dictionary.save('yelp.dict')
else:
    dictionary = gensim.corpora.Dictionary.load(dictpath)

train_texts = tokens
train_labels = labels

# compute the max sequence length
# why do we need to do that? - to ensure the number of words is less than the number of input nodes in the CNN
lengths = list(map(lambda x: len(x), train_texts))
a = np.array(lengths)
MAX_SEQUENCE_LENGTH = int(np.percentile(a, SEQUENCE_LENGTH_PERCENTILE))
# convert all texts to dictionary indices
train_texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
# pad or truncate the texts
x_data = pad_sequences(train_texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))
y_data = train_labels

# building cnn
model = Sequential()

model.add(Embedding(len(dictionary)+1,
                    EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH))
# add drop out for the input layer, why do you think this might help?
# drop out typical prevents overfitting, this will help the model learn for instances
# when certain words are missing
model.add(Dropout(dropout_rate))

# add a 1 dimensional conv layer
# a rectified linear activation unit, returns input if input > 0 else 0
model.add(Conv1D(filters=n_filters,
                  kernel_size=window_size,
                  activation='relu'))

# add a max pooling layer
model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
model.add(Flatten())

# add 0 or more fully connected layers with drop out
for _ in range(n_layers):
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units,
                    activation=dense_activation,
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

# add last fully connected layer with softmax activation
model.add(Dropout(dropout_rate))
model.add(Dense(len(train_labels[0]),
                activation='softmax',
                kernel_regularizer=l2(l2_penalty),
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'))

# compile model and specify optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# train model with early stopping
early_stopping = EarlyStopping(patience=patience)
Y = np.array(y_data)

fit = model.fit(x_data, Y,
                batch_size=batch_size, epochs=epochs,
                validation_split=VALIDATION_SPLIT,
                verbose=1,
                callbacks=[early_stopping])

print(fit.history['acc'][-1])

if model_file:
    model.save(model_file)
