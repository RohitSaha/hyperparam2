from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import time

sess=tf.Session()
new_saver = tf.train.import_meta_graph('my_test_model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

# layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
# layer_1 = tf.nn.relu(layer_1)
#
# layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
# layer_2 = tf.nn.relu(layer_2)
#
# out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=math.exp(-2)).minimize(cost)
