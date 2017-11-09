from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import sys

# PARAMETERS
maxlen = 40  # length of sentences
step = 20  # stride to cut data at
LSTMsize = 16  # size of LSTM layer
densesize = 32  # size of dense layer

# PROCESS DATA

path_wiki = "full-simple-wiki.txt"
text_wiki = open(path_wiki, encoding='utf8').read().lower()
path_tay = "lyrics.txt"
text_tay = open(path_tay, encoding='utf8').read().lower()
print('simple wiki length:', len(text_wiki))
print('taylor swift length:', len(text_tay))

chars = sorted(list(set(text_wiki).union(set(text_tay))))
numchars = len(chars)  # number of characters
print('total chars:', numchars)
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
xwiki, ywiki = vectorize_corpus(text_wiki, maxlen, step, len(chars), char_indices)

# TF LTSM

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200
num_input = numchars

# tf Graph input
X = tf.placeholder("float", [None, maxlen, num_input])
Y = tf.placeholder("float", [None, numchars])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([LSTMsize, numchars]))
}
biases = {
    'out': tf.Variable(tf.random_normal([numchars]))
}

print(X)
print(Y)
print(weights['out'])
print(biases['out'])


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, maxlen, n_input)
    # Required shape: 'maxlen' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'maxlen' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, maxlen, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(LSTMsize, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, maxlen, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, maxlen, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))