'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# PARAMETERS
maxlen = 40  # length of sentences
step_song = 200  # stride to cut song data at
step_tay = 3  # stride to cut tay data at
LSTMsize = 16  # size of LSTM layer
densesize = 32  # size of dense layer

path_song = "songdata_new.txt"
text_song = open(path_song, encoding='utf8').read().lower()
path_tay = "lyrics.txt"
text_tay = open(path_tay, encoding='utf8').read().lower()
print('simple song length:', len(text_song))
print('taylor swift length:', len(text_tay))

chars = sorted(list(set(text_song).union(set(text_tay))))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def vectorize_corpus(corpus, seq_len, step, num_chars, char_encoding):
    sentences = []
    next_chars = []
    for i in range(0, len(corpus) - seq_len, step):
        sentences.append(corpus[i: i + seq_len])
        next_chars.append(corpus[i + seq_len])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), seq_len, num_chars), dtype=np.bool)
    y = np.zeros((len(sentences), num_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_encoding[char]] = 1
        y[i, char_encoding[next_chars[i]]] = 1
    return x, y

# cut the text in semi-redundant sequences of maxlen characters
x, y = vectorize_corpus(text_song, maxlen, step_song, len(chars), char_indices)


# build the model: a single LSTM
print('Build model...')
def buildmodel(modeltype):
    trainable = modeltype == "song"
    model = Sequential()
    model.add(LSTM(LSTMsize, input_shape=(maxlen, len(chars)), trainable=trainable))
    model.add(Dense(densesize))
    model.add(Activation('relu'))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
    
model = buildmodel("song")
taymodel = buildmodel("tay")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train_and_sample(model, x, y, epochs, save_as, corpus, char_encoder, char_decoder):
    # train the model, output generated text after each iteration
    input_length = x.shape[1]
    num_chars = x.shape[2]
    for iteration in range(1, epochs+1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        # model.load_weights('./recurrent.h5')
        model.fit(x, y,
                  batch_size=128,
                  epochs=1)
        model.save(save_as)

        # when using spot instances on AWS: save repeatedly to a persistent volume, along with iteration number
        # if killed should be able to pick up where it left off based on the saved model

        start_index = random.randint(0, len(corpus) - input_length - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = corpus[start_index: start_index + input_length]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, input_length, num_chars))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_encoder[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = char_decoder[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

train_and_sample(model, x, y, 10, 'recurrent.h5', text_song, char_indices, indices_char)
for songlayer, taylayer in zip(model.layers, taymodel.layers):
    taylayer.set_weights(songlayer.get_weights())

# free memory
del x
del y
del text_song

x, y = vectorize_corpus(text_tay, maxlen, step_tay, len(chars), char_indices)
train_and_sample(model, x, y, 10, 'tay.h5', text_tay, char_indices, indices_char)
