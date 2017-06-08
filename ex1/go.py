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

DRY_RUN = False
DRY_RUN_TRAIN_EPOCHS = 2
TRAIN_EPOCHS = 5000


def build_model(X, learning_rate=0.01, beta1=0.8, beta2=0.999, n_hidden_1=700,n_hidden_2=512, n_output=30, keep_prob_lay1=0.5, keep_prob_rest=0.8, beta=0.00001):
    # tf Graph input (only pictures)

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
    activation_layer = tf.nn.sigmoid
    #activation_layer = tf.nn.relu          #TODO: shouldn't RELU be better?

    enc_layer_1 = activation_layer(tf.add(tf.matmul(X, weights['encoder_h1']),biases['encoder_b1']))
    enc_layer_dropout_1 = tf.nn.dropout(enc_layer_1, keep_prob_lay1)
    enc_layer_2 = activation_layer(tf.add(tf.matmul(enc_layer_dropout_1, weights['encoder_h2']),biases['encoder_b2']))
    enc_layer_dropout_2 = tf.nn.dropout(enc_layer_2, keep_prob_rest)
    enc_layer_3 = activation_layer(tf.add(tf.matmul(enc_layer_dropout_2, weights['encoder_h3']),biases['encoder_b3']))
    enc_layer_dropout_3 = tf.nn.dropout(enc_layer_3, keep_prob_rest)
    encoder_op = enc_layer_dropout_3


    # Building the decoder
    decd_layer_1 = activation_layer(tf.add(tf.matmul(encoder_op, weights['decoder_h1']), biases['decoder_b1']))
    decd_layer_2 = activation_layer(tf.add(tf.matmul(decd_layer_1, weights['decoder_h2']), biases['decoder_b2']))
    decd_layer_3 = activation_layer(tf.add(tf.matmul(decd_layer_2, weights['decoder_h3']), biases['decoder_b3']))
    decoder_op = decd_layer_3

    # L2 penalty - encoder only
    # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
    regulize_enc_l2 = tf.nn.l2_loss(weights['encoder_h1']) + tf.nn.l2_loss(weights['encoder_h2']) + tf.nn.l2_loss(weights['encoder_h3'])

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X


    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    total_loss = (cost + beta * regulize_enc_l2)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7, beta2=beta2).minimize(total_loss)

    return encoder_op, decoder_op,  optimizer, total_loss, cost, regulize_enc_l2


def main():
    print "Hello world!"
    training_epochs = TRAIN_EPOCHS

    print "test num_examples: %d; validation num_examples: %d; train num_examples: %d" % (mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)
    #test num_examples: 55000; validation num_examples: 5000; train num_examples: 10000


    if DRY_RUN:
        training_epochs = DRY_RUN_TRAIN_EPOCHS
        print "DRY_RUN; set training_epochs to %d" % (training_epochs)

    batch_size_test = 1024
    batch_size_options = [256, 16, 32, 64, 128, 512]
    learning_rate_options = [0.0004, 0.001, 0.005, 0.05, 0.1, 0.01, 0.003,0.008, 0.0001]

    X = tf.placeholder("float", [None, n_input])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb
    #distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    #pred = tf.arg_min(distance, 0)
    # tf Graph Input
    #xtr = tf.placeholder("float", [None, 784])
    xtr = tf.placeholder("float", [None, 30])
    xte = tf.placeholder("float", [30])

    for batch_size in batch_size_options:
        for learning_rate in learning_rate_options:
            display_step = 5

            n_hidden_1 = 512
            n_hidden_2 = 128
            n_output = 30

            desc = "bs_%d_lr_%2.5f_hs_%d_%d_out_%d" % (batch_size, learning_rate, n_hidden_1, n_hidden_2, n_output)


            # Launch the graph
            with tf.Session() as sess:

                encoder_op, decoder_op, optimizer, total_loss, cost, regulize_enc_l2 = build_model(X=X, learning_rate=learning_rate,
                  n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
                  n_output=n_output)

                # Initializing the variables
                init = tf.global_variables_initializer()

                tf.summary.scalar('cost', cost)
                tf.summary.scalar('total_loss', total_loss)
                tf.summary.scalar('regulize_enc_l2', regulize_enc_l2)
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('batch_size', batch_size)

                summaries = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(logs_folder, desc + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")))
                writer.add_graph(sess.graph)

                sess.run(init)
                total_batch = int(mnist.train.num_examples/batch_size)

                # Training cycle
                for epoch in range(training_epochs):
                    # Loop over all batches
                    for i in range(total_batch):
                        if (epoch % display_step) == 0 or (epoch + 1 == training_epochs):
                            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                            _, c, tl, enc_data, decd_data, summary = sess.run([optimizer, cost, total_loss, encoder_op, decoder_op, summaries], feed_dict={X: batch_xs})
                            writer.add_summary(summary, epoch * total_batch + i)


                    # Display logs per epoch step
                    if (epoch % display_step) == 0 or (epoch+1 == training_epochs):
                        #print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)
                        print "Epoch: %05d cost: %.9f   total_loss: %.9f" % (epoch+1, c, tl)

                        #batch_size_test
                        test_batch_xs, test_batch_ys = mnist.test.next_batch(batch_size_test)
                        test_enc_data = sess.run([encoder_op], feed_dict={X: test_batch_xs})

                        correct = 0
                        accuracy = 0
                        # Calculate L1 Distance
                        distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
                        pred_nn = tf.arg_min(distance, 0)
                        pred_nn_k2 = tf.nn.top_k(tf.negative(distance), k=2)        # top 2 nearest neighbors


                        for i in range(len(test_enc_data[0])):
                            # Get nearest neighbor
                            nn_index, nn_index2 = sess.run([pred_nn, pred_nn_k2], feed_dict={xtr: test_enc_data[0], xte:test_enc_data[0][i, :]})
                            nearest_neighbour_first_neigh = nn_index2[1][0]
                            nearest_neighbour_second_neigh = nn_index2[1][1]
                            assert nn_index == nearest_neighbour_first_neigh    # "nearest neighbor" is the point itself! since it's included in the search

                            # Calculate accuracy
                            if np.argmax(test_batch_ys[nearest_neighbour_second_neigh]) == np.argmax(test_batch_ys[i]):
                            #if np.argmax(test_batch_ys[nn_index]) == np.argmax(test_batch_ys[i]):
                                #accuracy += 1. / len(Xte)
                                correct += 1

                        accuracy = 100.0 * correct / len(test_batch_ys)
                        print "correct %d/%d (%2.5f%%)" % (correct, len(test_batch_ys), accuracy)

                print("done training; Optimization Finished! trained: %s" % (desc))


if __name__ == '__main__':
    main()