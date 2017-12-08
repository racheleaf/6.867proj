import numpy as np
import random

import tqdm
import unidecode

import torch
from torch import nn
from torch.autograd import Variable

from torchlstm import freeze_layer, TextLSTM
from torchconfig import config
from torchgen import generate


def read_file(filename):
    with open(filename) as ifile:
        return unidecode.unidecode(ifile.read())


def random_training_set(corpus, chunk_len, batch_size, char_encoding):
    x = torch.LongTensor(batch_size, chunk_len)
    y = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, len(corpus) - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = corpus[start_index:end_index]
        for t in range(len(chunk) - 1):
            x[bi, t] = char_encoding[chunk[t]]
            y[bi, t] = char_encoding[chunk[t+1]]
    x = Variable(x)
    y = Variable(y)
    if config['cuda']:
        x = x.cuda()
        y = y.cuda()
    return x, y


def vectorize_corpus(corpus, seq_len, step, num_chars, char_encoding):
    sentences = []
    for i in range(0, len(corpus) - seq_len, step):
        sentences.append(corpus[i: i + seq_len])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), seq_len, num_chars), dtype=np.bool)
    y = np.zeros((len(sentences), seq_len, num_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t in range(len(sentence) - 1):
            x[i, t, char_encoding[sentence[t]]] = 1
            y[i, t, char_encoding[sentence[t+1]]] = 1
    x = Variable(x)
    y = Variable(y)
    if config['cuda']:
        x = x.cuda()
        y = y.cuda()
    return x, y


def buildmodel(modeltype):
    model = TextLSTM(len(chars), LSTMsize, len(chars), 1)
    if config['cuda']:
        model = model.cuda()
    return model


def freeze_codec(network):
    freeze_layer(network.encoder)
    freeze_layer(network.decoder)


# TODO: rewrite for torch
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_step(model, optimizer, inp, target, loss_function, chunk_len, batch_size):
    hidden = model.init_hidden(batch_size)
    if config['cuda']:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    model.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = model(inp[:, c], hidden)
        loss += loss_function(output.view(batch_size, -1), target[:, c])

    loss.backward()
    optimizer.step()
    return loss.data[0] / chunk_len


def train_and_sample(model, epochs, save_as, corpus, char_encoder, char_decoder):
    # train the model, output generated text after each iteration
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    chunk_len = config['maxlen']
    batch_size = config['batchsize']
    for iteration in tqdm.tqdm(range(1, epochs+1)):
        # TODO: train
        x, y = random_training_set(corpus, config['maxlen'], config['batchsize'], char_encoder)
        loss = train_step(model, optimizer, x, y, criterion, chunk_len, batch_size)
        print(f'Iteration {iteration}: loss = {loss}')
        # model.load_weights('./recurrent.h5')
        if iteration % 10 == 0:
            torch.save(model, save_as)
        if iteration % 100 == 0:
            seed = corpus[random.randint(0, len(corpus) - 1)]
            print('Sampling with seed \'{}\''.format(seed))
            for temp in [0.2, 0.5, 1, 1.2]:
                print('Temperature: {}'.format(temp))
                print(generate(model, 200, seed, temp, char_encoder, char_decoder))

        # when using spot instances on AWS: save repeatedly to a persistent volume, along with iteration number
        # if killed should be able to pick up where it left off based on the saved model


# PARAMETERS
maxlen = config['maxlen']  # length of sentences
LSTMsize = config['lstmsize']  # size of LSTM layer
epochs = config['epochs']

path_song = "songdata_new.txt"
text_song = read_file(path_song)
path_tay = "lyrics.txt"
text_tay = read_file(path_tay)
print('simple song length:', len(text_song))
print('taylor swift length:', len(text_tay))

chars = sorted(list(set(text_song).union(set(text_tay))))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build the model: a single LSTM
print('Build model...')
model = buildmodel("song")


train_and_sample(model, epochs, 'recurrent.h5', text_song, char_indices, indices_char)
freeze_codec(model)

# free memory
del text_song

# x, y = vectorize_corpus(text_tay, maxlen, step_tay, len(chars), char_indices)
train_and_sample(model, epochs, 'tay.h5', text_tay, char_indices, indices_char)
