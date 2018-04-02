import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.layers import LSTM, Input, Dot, Softmax, Multiply, Concatenate, Subtract, Dense, Lambda, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model, load_model
from os.path import isfile

SentenceLen = 100
WordVecLen = 300

if not isfile('NLI.h5'):
    raise ValueError('Weights not available. Please run train.ipynb.')

if not isfile('tokenizer.pickle'):
    raise ValueError('Tokenizer not available. Please run train.ipynb.')
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

if not isfile('embeddings.npy'):
    raise ValueError('Weights not available. Please run train.ipynb.')

def load_embeddings():
    embedding_matrix = np.load('embeddings.npy')
    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                WordVecLen,
                                weights=[embedding_matrix],
                                input_length=SentenceLen,
                                trainable=False)
    return embedding_layer

embedding_layer = load_embeddings()

def expand_rep(x, r, a):
    y = K.expand_dims(x, axis=a)
    y = K.repeat_elements(y, r, axis=a)
    return y

bilstm1 = Bidirectional(LSTM(300, return_sequences=True))
bilstm2 = Bidirectional(LSTM(300, return_sequences=True))

i1 = Input(shape=(SentenceLen,), dtype='float32')
i2 = Input(shape=(SentenceLen,), dtype='float32')

x1 = embedding_layer(i1)
x2 = embedding_layer(i2)

x1 = bilstm1(x1)
x2 = bilstm1(x2)

e = Dot(axes=2)([x1, x2])
e1 = Softmax(axis=2)(e)
e2 = Softmax(axis=1)(e)
e1 = Lambda(expand_rep, arguments={'r' : 2 * WordVecLen, 'a' : 3})(e1)
e2 = Lambda(expand_rep, arguments={'r' : 2 * WordVecLen, 'a' : 3})(e2)

_x1 = Lambda(expand_rep, arguments={'r' : K.int_shape(x1)[1], 'a' : 1})(x2)
_x1 = Multiply()([e1, _x1])
_x1 = Lambda(K.sum, arguments={'axis' : 2})(_x1)
_x2 = Lambda(expand_rep, arguments={'r' : K.int_shape(x2)[1], 'a' : 2})(x1)
_x2 = Multiply()([e2, _x2])
_x2 = Lambda(K.sum, arguments={'axis' : 1})(_x2)

m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

y1 = bilstm2(m1)
y2 = bilstm2(m2)

mx1 = Lambda(K.max, arguments={'axis' : 1})(y1)
av1 = Lambda(K.mean, arguments={'axis' : 1})(y1)
mx2 = Lambda(K.max, arguments={'axis' : 1})(y2)
av2 = Lambda(K.mean, arguments={'axis' : 1})(y2)

y = Concatenate()([av1, mx1, av2, mx2])
y = Dense(1024, activation='tanh')(y)
y = Dense(3, activation='softmax')(y)

model = Model(inputs=[i1, i2], outputs=y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('NLI.h5')

def PadSeq(text):
    sequences = tokenizer.texts_to_sequences(text)
    return pad_sequences(sequences, maxlen=SentenceLen)

print()
print()
print('Model loaded successfully.')

while True:
    print()
    p = PadSeq([input('Input premise\t\t: ')])
    h = PadSeq([input('Input hypothesis\t: ')])

    pred = model.predict([p, h])
    pred = np.argmax(pred)

    res = '-'
    if pred == 0:
        res = 'Entailment'
    elif pred == 1:
        res = 'Contradiction'
    elif pred == 2:
        res = 'Neutral'

    print()
    print('The given premise-hypothesis pair is a/an %s' % (res))