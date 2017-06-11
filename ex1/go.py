#!/usr/bin/python
# based on:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb

# TODOs:
# http://sebastianruder.com/optimizing-gradient-descent/
# Add noise

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
import time
from sklearn.manifold import TSNE


mnist = None
logs_folder = "logs"
models_folder = "models"
n_input = 784  # MNIST data input (img shape: 28*28)

DRY_RUN = False
DRY_RUN_TRAIN_EPOCHS = 3
DRY_RUN_DISPLAY_STEP = 10
DRY_RUN_MIN_SAVE_ACC_PER = 60
DISPLAY_STEP_NORMAL = 50
TRAIN_EPOCHS_NORMAL = 5000
MIN_SAVE_ACC_PER_NORMAL = 92

parameter_search = False

min_save_acc_per = MIN_SAVE_ACC_PER_NORMAL
if DRY_RUN:
    min_save_acc_per = DRY_RUN_MIN_SAVE_ACC_PER


def build_model(desc, X, weight_init="xv", keep_prob_lay1=0.8, keep_prob_rest=0.5, learning_rate=0.01, beta1=0.8, beta2=0.999, n_hidden_1=700,n_hidden_2=512, n_output=30, beta=0.000001):
    # https://stackoverflow.com/a/36784797

    with tf.variable_scope(desc):
        def get_variable(name, weight_init_func, shape):
            if weight_init_func == tf.random_normal:
                return tf.Variable(weight_init_func(shape))
            elif weight_init_func == tf.contrib.layers.variance_scaling_initializer:
                return tf.get_variable(name, shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer())

        if weight_init == 'weight_init':
            weight_init_func = tf.contrib.layers.variance_scaling_initializer
        else:
            weight_init_func = tf.random_normal

        list_weights = list()
        list_weights.append(['encoder_h1', weight_init_func, [n_input, n_hidden_1] ])
        list_weights.append(['encoder_h2', weight_init_func, [n_hidden_1, n_hidden_2] ])
        list_weights.append(['encoder_h3', weight_init_func, [n_hidden_2, n_output] ])
        list_weights.append(['decoder_h1', weight_init_func, [n_output, n_hidden_2] ])
        list_weights.append(['decoder_h2', weight_init_func, [n_hidden_2, n_hidden_1] ])
        list_weights.append(['decoder_h3', weight_init_func, [n_hidden_1, n_input] ])

        list_biases = list()
        list_biases.append(['encoder_b1', weight_init_func, [n_hidden_1] ])
        list_biases.append(['encoder_b2', weight_init_func, [n_hidden_2] ])
        list_biases.append(['encoder_b3', weight_init_func, [n_output] ])
        list_biases.append(['decoder_b1', weight_init_func, [n_hidden_2] ])
        list_biases.append(['decoder_b2', weight_init_func, [n_hidden_1] ])
        list_biases.append(['decoder_b3', weight_init_func, [n_input] ])

        weights = {}
        for weight in list_weights:
            weights[weight[0]] = get_variable(weight[0], weight[1], weight[2])

        biases = {}
        for bias in list_biases:
            biases[bias[0]] = get_variable(bias[0], bias[1], bias[2])


        # Building the encoder
        activation_layer = tf.nn.sigmoid
        #activation_layer = tf.nn.tanh
        #activation_layer = tf.nn.relu          #TODO: shouldn't RELU be better?


        X_after_dropout = tf.nn.dropout(X, keep_prob_lay1)
        enc_layer_1 = activation_layer(tf.add(tf.matmul(X_after_dropout, weights['encoder_h1']),biases['encoder_b1']))
        enc_layer_dropout_1 = tf.nn.dropout(enc_layer_1, keep_prob_rest)
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

        y_pred = decoder_op
        #y_true = X_after_dropout
        y_true = X


        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        total_loss = (cost + beta * regulize_enc_l2)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7, beta2=beta2).minimize(total_loss)
        return encoder_op, decoder_op,  optimizer, total_loss, cost, regulize_enc_l2


def mnist_load():
    global mnist
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def mnist_normalize_global():
    def normalize_global(images, train_mean=None, train_sd=None):
        if train_mean is None or train_sd is None:
            train_mean, train_sd = np.mean(images), np.std(images)
        images_normalized = images - train_mean
        images_normalized = images_normalized / train_sd
        return images_normalized, train_mean, train_sd

    global mnist

    # normalization train, validation, test images
    mnist.train._images, train_mean, train_sd = normalize_global(mnist.train.images)
    mnist.validation._images, _, _ = normalize_global(mnist.validation.images, train_mean, train_sd)
    mnist.test._images, _, _ = normalize_global(mnist.test.images, train_mean, train_sd)


def mnist_normalize_local():
    def normalize_local(images):
        for index, image in enumerate(images):
            m, sd = np.mean(image), np.std(image)
            #print "image m, sd: " , m, sd
            image = image - m
            image = image / sd
            images[index] = image
        return images

    global mnist

    # _images breaks encapsulation, but other options are to re-write/change code
    mnist.train._images = normalize_local(mnist.train.images)
    mnist.validation._images = normalize_local(mnist.validation.images)
    mnist.test._images = normalize_local(mnist.test.images)

    print "done with mnist_normalize_local"


def mnist_normalize():
    mnist_normalize_local()
    #mnist_normalize_global()


def get_knn_error(sess, encoder_op, X, keep_prob_lay1, keep_prob_rest, n_output):
    global mnist

    # Nearest Neighbor calculation using L1 Distance
    xtr = tf.placeholder("float", [None, n_output])
    xte = tf.placeholder("float", [n_output])

    test_batch_xs, test_batch_ys = mnist.test.next_batch(mnist.test.num_examples)
    test_enc_data = sess.run([encoder_op],
                             feed_dict={X: test_batch_xs, keep_prob_lay1: 1, keep_prob_rest: 1})

    correct = 0
    accuracy = 0

    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    pred_nn = tf.arg_min(distance, 0)
    pred_nn_k2 = tf.nn.top_k(tf.negative(distance), k=2)  # get top 2 nearest neighbors

    for i in range(len(test_enc_data[0])):
        # Get nearest neighbor
        nn_index, nn_index2 = sess.run([pred_nn, pred_nn_k2],
                                       feed_dict={xtr: test_enc_data[0], xte: test_enc_data[0][i, :]})
        nearest_neighbour_first_neigh = nn_index2[1][0]
        nearest_neighbour_second_neigh = nn_index2[1][1]
        assert nn_index == nearest_neighbour_first_neigh  # "nearest neighbor" is the point itself! since it's included in the search

        # Calculate accuracy
        if np.argmax(test_batch_ys[nearest_neighbour_second_neigh]) == np.argmax(test_batch_ys[i]):
            correct += 1

    accuracy = 100.0 * correct / len(test_batch_ys)
    return correct, len(test_batch_ys), accuracy, test_enc_data, test_batch_ys,  test_batch_xs

# Scale and visualize the embedding vectors
# Based on: http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
def plot_embedding(X, Y, images, save_path, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(np.argmax(Y[i])),
                 color=plt.cm.Set1(np.argmax(Y[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if False:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                img1 = np.reshape(images[i], [28, 28])
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(img1, cmap=plt.cm.gray_r),
                    X[i],
                    xybox=(10., -10.)
                )
                ax.add_artist(imagebox)

    plt.savefig(save_path + '.jpg', dpi=1000)
    plt.savefig(save_path + '.pdf')
    #plt.show()



def main():
    global mnist

    print "Hello world!"
    training_epochs = TRAIN_EPOCHS_NORMAL
    display_step = DISPLAY_STEP_NORMAL

    if DRY_RUN:
        training_epochs = DRY_RUN_TRAIN_EPOCHS
        display_step = DRY_RUN_DISPLAY_STEP
        print "DRY_RUN; set training_epochs to %d; display_step to %d" % (training_epochs, display_step)

    mnist_load()
    mnist_normalize()


    print "test num_examples: %d; validation num_examples: %d; train num_examples: %d" % (mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)

    batch_size_test = 10000
    if parameter_search:
        batch_size_options = [64, 128, 256, 16, 32, 512]
        learning_rate_options = [0.01, 0.008, 0.007, 0.004] #0.007, 0.0003, 0.001, 0.005, 0.05, 0.003,0.008, 0.0001]
    else:
        batch_size_options = [64]
        learning_rate_options = [0.004]


    n_hidden_1 = 256
    n_hidden_2 = 128
    n_output = 30
    keep_prob_lay1_val = 0.8
    keep_prob_rest_val = 1
    weight_init = 'xv'

    X = tf.placeholder("float", [None, n_input])
    keep_prob_lay1 = tf.placeholder(tf.float32)
    keep_prob_rest = tf.placeholder(tf.float32)


    for batch_size in batch_size_options:
        for learning_rate in learning_rate_options:

            desc = "lr_%2.5f_kp1_%0.2f_kpr_%0.2f_wi_%s_bs_%d_hs_%d_%d_out_%d" % (learning_rate, keep_prob_lay1_val, keep_prob_rest_val, weight_init, batch_size, n_hidden_1, n_hidden_2, n_output)
            print "start trainign desc: ", desc

            # Launch the graph
            with tf.Session() as sess:
                encoder_op, decoder_op, optimizer, total_loss, cost, regulize_enc_l2 = \
                    build_model(desc=desc, X=X, weight_init='xv', keep_prob_lay1=keep_prob_lay1_val, keep_prob_rest=keep_prob_rest_val,
                        learning_rate=learning_rate,
                        n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
                        n_output=n_output)

                # Initializing the variables
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()

                tf.summary.scalar('cost', cost)
                tf.summary.scalar('total_loss', total_loss)
                tf.summary.scalar('regulize_enc_l2', regulize_enc_l2)

                summaries = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(logs_folder, desc + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")))
                writer.add_graph(sess.graph)
                sess.run(init)

                # Training cycle
                total_batch = int(mnist.train.num_examples/batch_size)
                for epoch in range(training_epochs):
                    # Loop over all batches
                    for i in range(total_batch):
                        #if (epoch % display_step) == 0 or (epoch + 1 == training_epochs):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        _, c, tl, enc_data, decd_data, summary = sess.run(
                            [optimizer, cost, total_loss, encoder_op, decoder_op, summaries],
                            feed_dict={X: batch_xs, keep_prob_lay1: keep_prob_lay1_val, keep_prob_rest: keep_prob_rest_val})
                        writer.add_summary(summary, epoch * total_batch + i)



                    if epoch % 10 == 0:
                        batch_xs_val, batch_ys_val = mnist.validation.next_batch(mnist.validation.num_examples)
                        c_val, tl_val = sess.run(
                            [cost, total_loss],
                            feed_dict={X: batch_xs_val, keep_prob_lay1: keep_prob_lay1_val,
                            keep_prob_rest: keep_prob_rest_val})

                        print "Epoch: %05d mini-batch cost:    %.9f   mini-batch total_loss:    %.9f" % (epoch + 1, c, tl)
                        print "Epoch: %05d validation cost:    %.9f   validation total_loss:    %.9f" % (epoch+1, c_val, tl_val)

                    # Display logs per epoch step
                    if (epoch % display_step) == 0 or (epoch+1 == training_epochs):
                        correct, total, accuracy, enc_data, valida_batch_ys, valida_batch_xs = get_knn_error(sess=sess, encoder_op=encoder_op, X=X, keep_prob_lay1=keep_prob_lay1,
                            keep_prob_rest=keep_prob_rest, n_output=n_output)

                        print "validation KNN: correct %d/%d (%2.5f%%)" % (correct, total, accuracy)

                        if (accuracy > min_save_acc_per):
                            curr_model_desc = desc + "epc_%d_acc_%2.3f_test" % (epoch, accuracy)
                            save_path = saver.save(sess, os.path.join(models_folder, curr_model_desc + ".ckpt"))
                            print "model %s with %2.3f%% accuracy saved to %s" % (desc, accuracy, save_path)

                            fig_save_path = os.path.join(models_folder, curr_model_desc)
                            tsne_model = TSNE(n_components=2, random_state=0)
                            #X_tsne = tsne_model.fit_transform(valida_batch_xs)
                            X_tsne = tsne_model.fit_transform(enc_data[0])
                            plot_embedding(X_tsne, valida_batch_ys, valida_batch_xs, fig_save_path,
                               "t-SNE embedding of the digits for model %s" % desc)


                print("done training; Optimization Finished! trained: %s" % (desc))


if __name__ == '__main__':
    main()

