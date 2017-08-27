#coding: utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import input_data

num_units = 128
layer = 3
step = 10000

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(shape=shape, value=0.1)
    return tf.Variable(initial)

def lstm():

    mnist = input_data.read_data_sets("../data/", one_hot=True)

    sess = tf.InteractiveSession()

    x = tf.placeholder("float", [None, 28, 28])
    y = tf.placeholder("float", [None, 10])

    x_ = tf.unstack(x, 28 , 1)

    weight = weight_variable([num_units, 10])
    bias = weight_variable([10])

    multi_cell = []

    for i in range(layer):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
        multi_cell.append(lstm_cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(multi_cell)

    output, state = rnn.static_rnn(cell, x_, dtype=tf.float32)

    pred = tf.matmul(output[-1], weight) + bias

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimzer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

    sess.run(tf.initialize_all_variables())

    for i in range(step):
        batch = mnist.train.next_batch(50)

        batch_x = batch[0].reshape(-1, 28, 28)
        batch_y = batch[1]

        optimzer.run(feed_dict={x: batch_x, y: batch_y})

    test_x = mnist.test.images.reshape(-1, 28, 28)
    test_y = mnist.test.labels
    print accuracy.eval(feed_dict={x: test_x, y:test_y})

if __name__ == "__main__":
    lstm()