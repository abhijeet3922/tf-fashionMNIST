#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from dataset import fashion_MNIST
from util import config

def model(batch_x):

    b1 = tf.get_variable("b1", [config.n_hidden1], initializer = tf.zeros_initializer())
    h1 = tf.get_variable("h1", [config.n_input, config.n_hidden1],
                         initializer = tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.add(tf.matmul(batch_x,h1),b1))

    b2 = tf.get_variable("b2", [config.n_hidden2], initializer = tf.zeros_initializer())
    h2 = tf.get_variable("h2", [config.n_hidden1, config.n_hidden2],
                         initializer = tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,h2),b2))

    b3 = tf.get_variable("b3", [config.n_class], initializer = tf.zeros_initializer())
    h3 = tf.get_variable("h3", [config.n_hidden2, config.n_class],
                         initializer = tf.contrib.layers.xavier_initializer())

    layer3 = tf.add(tf.matmul(layer2,h3),b3)

    return layer3


def compute_loss(predicted, actual):

    total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = predicted,labels = actual)
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss


def create_optimizer():

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    return optimizer


def one_hot(n_class, Y):
    return np.eye(n_class)[Y]


def train(X_train, X_val, X_test, y_train, y_val, y_test, verbose = False):

    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")

    logits = model(X)
    avg_loss = compute_loss(logits, Y)
    optimizer = create_optimizer().minimize(avg_loss)
    validation_loss = compute_loss(logits, Y)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(config.n_epoch):

            epoch_loss = 0.
            num_batches = np.round(X_train.shape[0]/config.batch_size).astype(int)

            for i in range(num_batches):

                batch_X = X_train[(i*config.batch_size):((i+1)*config.batch_size),:]
                batch_y = y_train[(i*config.batch_size):((i+1)*config.batch_size),:]

                _, batch_loss = sess.run([optimizer, avg_loss],
                                                       feed_dict = {X: batch_X, Y:batch_y})

                epoch_loss += batch_loss

            epoch_loss = epoch_loss/num_batches

            val_loss = sess.run(validation_loss, feed_dict = {X: X_val ,Y: y_val})

            if verbose:
                print("epoch:{epoch_num}, train_loss: {train_loss}, train_accuracy: {train_acc}, val_loss: {valid_loss}, val_accuracy: {val_acc} ".format(
                                                       epoch_num = epoch,
                                                       train_loss = round(epoch_loss,3),
                                                       train_acc = round(float(accuracy.eval({X: X_train, Y: y_train})),2),
                                                       valid_loss = round(float(val_loss),3),
                                                       val_acc = round(float(accuracy.eval({X: X_val, Y: y_val})),2)
                                                      ))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))
        sess.close()


def main(_):

    fashion_mnist = fashion_MNIST.Dataset(data_download_path='../data/fashion', validation_flag=True, verbose=True)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fashion_mnist.load_data()

    y_train =  one_hot(config.n_class, Y_train)
    y_val = one_hot(config.n_class, Y_val)
    y_test = one_hot(config.n_class, Y_test)

    train(X_train, X_val, X_test, y_train, y_val, y_test, True)



if __name__ == '__main__' :
    tf.app.run(main)
