from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import time

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

with tf.Session() as sess:

    weights = {
        'h1': tf.Variable(tf.random_normal([784, 20])),
        'h2': tf.Variable(tf.random_normal([20, 30])),
        'out': tf.Variable(tf.random_normal([30, 10])),
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([20])),
        'h2': tf.Variable(tf.random_normal([30])),
        'out': tf.Variable(tf.random_normal([10])),
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=math.exp(-2)).minimize(cost)

    sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver()
    # saver.save(sess, 'my_test_model')
    batch_size = 100

    for i in range(2):
        num_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    print acc
