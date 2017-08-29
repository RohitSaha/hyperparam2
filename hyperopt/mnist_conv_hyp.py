# conn = Connection(client_token='TZCLEDVNUXIYPTVQRJTTKYYUZWKBPHCNOKIAHHQORJQHNKRG')
# experiment = conn.experiments().create(
#     name='Multi-Layer Perceptron',
#     parameters=[
#         {
#             'name': 'num_hidden_1',
#             'type': 'int',
#             'bounds': {
#                 'min': 10,
#                 'max': 784,
#             },
#         },
#         #{
#         #     'name': 'activation_1',
#         #     'type': 'categorical',
#         #     'categorical_values': [
#         #         "relu",
#         #         "sigmoid",
#         #         "tanh",
#         #     ]
#         # },
#         {
#             'name': 'num_hidden_2',
#             'type': 'int',
#             'bounds': {
#                 'min': 10,
#                 'max': 784,
#             },
#         },
#         # {
#         #     'name': 'activation_2',
#         #     'type': 'categorical',
#         #     'categorical_values': [
#         #         "relu",
#         #         "sigmoid",
#         #         "tanh",
#         #     ]
#         # },
#         # {
#         #     'name': 'optimizer',
#         #     'type': 'categorical',
#         #     'categorical_values': [
#         #         "adam",
#         #         "rmsprop",
#         #         "gradient_descent",
#         #     ]
#         # },
#         {
#             'name': 'log_learning_rate',
#             'type': 'double',
#             'bounds': {
#                 'min': math.log(1e-8),
#                 'max': math.log(1),
#             },
#         },
#         {
#             'name': 'epochs',
#             'type': 'int',
#             'bounds': {
#                 'min': 5,
#                 'max': 20,
#             },
#         },
#     ],
#     observation_budget=70,
#     metadata={
#         'template': 'python_tf_mlp'
#     }
# )

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

nh1 = 20
nh2 = 20
llr = -5
eps = 5

def create_model(sess):
    weights = {
        'h1': tf.Variable(tf.random_normal([784, nh1])),
        'h2': tf.Variable(tf.random_normal([nh1, nh2])),
        'out': tf.Variable(tf.random_normal([nh2, 10])),
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([nh1])),
        'h2': tf.Variable(tf.random_normal([nh2])),
        'out': tf.Variable(tf.random_normal([10])),
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=math.exp(llr)).minimize(cost)

    sess.run(tf.global_variables_initializer())

    batch_size = 100

    for i in range(eps):
        num_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    return out_layer

def evaluate_model():
    with tf.Session() as sess:
        model = create_model(sess)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    return float(acc)
