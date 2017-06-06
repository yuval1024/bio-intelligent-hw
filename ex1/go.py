#!/usr/bin/python
# based on:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
logs_folder = "logs"
n_input = 784  # MNIST data input (img shape: 28*28)

DRY_RUN = True

def build_model(learning_rate=0.01, n_hidden_1=512,n_hidden_2=256,n_output=30):
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_output])),

        'decoder_h1': tf.Variable(tf.random_normal([n_output, n_hidden_2])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_output])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b3': tf.Variable(tf.random_normal([n_input])),
    }


    # Building the encoder
    enc_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']),biases['encoder_b1']))
    enc_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(enc_layer_1, weights['encoder_h2']),biases['encoder_b2']))
    enc_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(enc_layer_2, weights['encoder_h3']),biases['encoder_b3']))
    encoder_op = enc_layer_3

    # Building the decoder
    decd_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(enc_layer_3, weights['decoder_h1']), biases['decoder_b1']))
    decd_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(decd_layer_1, weights['decoder_h2']), biases['decoder_b2']))
    decd_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(decd_layer_2, weights['decoder_h3']), biases['decoder_b3']))
    decoder_op = decd_layer_3

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    return X, encoder_op, decoder_op, cost, optimizer


def main():
    print "Hello world!"

    training_epochs = 100
    if DRY_RUN:
        training_epochs = 2
        print "DRY_RUN; set training_epochs to %d" % (training_epochs)


    batch_size_options = [256, 16, 32, 64, 128, 512]
    learning_rate_options = [0.1, 0.01, 0.005, 0.001, 0.0001]

    for batch_size in batch_size_options:
        for learning_rate in learning_rate_options:
            display_step = 10

            n_hidden_1 = 512
            n_hidden_2 = 256
            n_output = 30

            desc = "bs_%d_lr_%2.5f_hs_%d_%d_out_%d" % (batch_size, learning_rate, n_hidden_1, n_hidden_2, n_output)
            X, encoder_op, decoder_op, cost, optimizer = build_model(learning_rate=learning_rate,
                    n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2, n_output=n_output)


            # Launch the graph
            with tf.Session() as sess:
                # Initializing the variables
                init = tf.global_variables_initializer()

                tf.summary.scalar('cost', cost)
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('batch_size', batch_size)

                summaries = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(logs_folder, desc + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")))
                writer.add_graph(sess.graph)

                sess.run(init)

                total_batch = int(mnist.train.num_examples/batch_size)

                print "examples: ", mnist.train.num_examples, " batch_size: ", batch_size
                print "total_batch: ", total_batch

                # Training cycle
                for epoch in range(training_epochs):
                    # Loop over all batches
                    for i in range(total_batch):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        _, c, enc_data, decd_data, summary = sess.run([optimizer, cost, encoder_op, decoder_op, summaries], feed_dict={X: batch_xs})

                        writer.add_summary(summary, epoch*total_batch + i)
                    # Display logs per epoch step
                    if (epoch % display_step) == 0 or (epoch+1 == training_epochs):
                        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)


                print("done training; Optimization Finished!")


if __name__ == '__main__':
    main()