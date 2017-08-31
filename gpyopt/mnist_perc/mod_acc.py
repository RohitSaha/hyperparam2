from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

def create_model(sess,prs):
    p0 = int(prs[0])
    p1 = int(prs[1])
    p3 = int(prs[3])

    print str(p0) + " " + str(p1) + " " + str(prs[2]) + " " + str(p3)

    weights = {
        'h1': tf.Variable(tf.random_normal([784, p0])),
        'h2': tf.Variable(tf.random_normal([p0, p1])),
        'out': tf.Variable(tf.random_normal([p1, 10])),
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([p0])),
        'h2': tf.Variable(tf.random_normal([p1])),
        'out': tf.Variable(tf.random_normal([10])),
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=math.exp(prs[2])).minimize(cost)

    sess.run(tf.global_variables_initializer())

    batch_size = 100

    for i in range(p3):
        num_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    return out_layer

def evaluate_model(prs):
    with tf.Session() as sess:
        model = create_model(sess,prs)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    print float(acc)

    return 1-float(acc)


print best
